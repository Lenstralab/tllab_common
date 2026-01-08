from __future__ import annotations

import logging
import os
import re
import tempfile
import warnings
from contextlib import ExitStack, redirect_stdout
from functools import wraps
from inspect import getfullargspec
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Sequence

from .misc import capture_stderr

with capture_stderr():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow

    tensorflow.get_logger().setLevel(logging.ERROR)

import numpy as np
import pandas
import polars as pl

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
import ray  # noqa

logging.getLogger("ray").setLevel(logging.ERROR)

with redirect_stdout(StringIO()):
    from cellpose import models
from csbdeep.utils import normalize
from laptrack import LapTrack, ParallelBackend
from ndbioimage import Imread
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed
from tiffwrite import FrameInfo, IJTiffFile, IJTiffParallel
from tqdm.auto import tqdm, trange

from .findcells import findcells
from .pytrackmate import trackmate_peak_import

try:
    from stardist.models import StarDist2D  # noqa
except ImportError:
    StarDist2D = None

try:
    import imagej
    import scyjava

    def kill_vm():
        if scyjava.jvm_started():
            scyjava.shutdown_jvm()

except ImportError:
    imagej = None
    scyjava = None

    def kill_vm():
        return


def label_dist(labels: np.ndarray, lbl: int, mask: np.ndarray = None) -> np.ndarray:
    """make an array with distances to the edge of the lbl in labels, negative outside, positive inside"""
    lbl_mask = labels == lbl
    dist = -distance_transform_edt(lbl_mask == 0)
    dist[dist < 0] += 1
    dist += distance_transform_edt(lbl_mask == 1)
    dist[(labels != lbl) * (labels != 0)] = -np.inf
    if mask is not None:
        dist[mask] = -np.inf
    return dist


def interp_label(
    t: int,
    ts: Sequence[int],
    labels: Sequence[np.ndarray],
    lbl: int,
    mask: np.ndarray = None,
) -> np.ndarray:
    """return a label field with lbl at time q interpolated from labels at times ts"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return lbl * (
            interp1d(
                ts,
                np.dstack([label_dist(label, lbl, mask) for label in labels]),
                fill_value=np.zeros_like(labels[0]),
                bounds_error=False,
            )(t)
            > 0
        )


class SwapLabels:
    def __init__(self, tracks: pandas.DataFrame | pl.DataFrame, min_frames: int = None) -> None:  # noqa
        if isinstance(tracks, pl.DataFrame):
            if min_frames:
                tracks = tracks.filter(pl.len().over("track_id") > min_frames)
            if not tracks.is_empty():
                d = {
                    u: i for i, u in enumerate(tracks.select(pl.col("track_id").unique()).sort("track_id").to_series())
                }
                tracks = tracks.with_columns(track_id=pl.col("track_id").replace_strict(d))
            self.tracks = tracks
        else:
            if min_frames:
                tracks = (
                    tracks.groupby("label")
                    .apply(lambda df: df.assign(n_frames=len(df)))
                    .query("n_frames > @min_frames", local_dict=dict(min_frames=min_frames))
                    .reset_index(drop=True)
                )
            # ensure that labels are consecutive
            if not tracks.empty:
                d = {u: i for i, u in enumerate(tracks["label"].unique(), 1)}
                tracks["label"] = tracks.apply(lambda s: d[s["label"]], 1)  # type: ignore
            self.tracks = tracks

    def __call__(self, frame_in: np.ndarray, t: int) -> np.ndarray:
        frame_out = np.zeros_like(frame_in)
        for i, j in self.get_mapping(t).items():
            frame_out[frame_in == i] = j
        return frame_out

    def get_mapping(self, t: int) -> dict[int, int]:
        if isinstance(self.tracks, pl.DataFrame):
            return {
                int(i): int(j)
                for i, j in self.tracks.filter(pl.col("t") == t).select("cell", "track_id").to_numpy()
                if np.isfinite(i)
            }
        else:
            return {
                int(i): int(j)
                for i, j in self.tracks.query("t == @t", local_dict=dict(t=t))[
                    ["median_intensity", "label"]
                ].to_numpy()
                if np.isfinite(i)
            }

    def get_labels(self) -> np.ndarray:
        if isinstance(self.tracks, pl.DataFrame):
            return self.tracks.select(pl.col("track_id").unique()).to_series().to_numpy()
        else:
            return self.tracks["label"].unique()


def sort_labels(tracks: pl.DataFrame | pandas.DataFrame) -> pl.DataFrame | pandas.DataFrame:
    """make labels consistent across different runs by sorting them according to mean cell area"""
    if isinstance(tracks, pl.DataFrame):
        relabel_dict = {
            int(key): int(value)
            for value, (key,) in enumerate(
                tracks.group_by("track_id").agg(pl.mean("area")).sort("area").select("track_id").iter_rows()
            )
        }
        return tracks.with_columns(track_id=pl.col("track_id").replace_strict(relabel_dict))
    else:
        relabel_dict = {
            int(key): value  # type: ignore
            for value, key in enumerate(tracks.groupby("label").aggregate("mean").sort_values("area").index, 1)
        }
        return tracks.groupby("label").apply(lambda x: x.assign(label=relabel_dict[x["label"].mean()]))  # type: ignore


def get_time_points(t: int, missing: Sequence[int]) -> tuple[int, int]:
    t_a = t - 1
    while t_a in missing:
        t_a -= 1
    t_b = t + 1
    while t_b in missing:
        t_b += 1
    return t_a, t_b


def interpolate_missing(tracks: pl.DataFrame | pandas.DataFrame, t_len: int = None) -> pl.DataFrame | pandas.DataFrame:
    """interpolate the position of the cell in missing frames"""
    missing = []
    if isinstance(tracks, pl.DataFrame):
        for (track_id,), group in tracks.group_by("track_id"):
            if t_len is None:
                t_missing = list(
                    set(
                        range(int(group.select(pl.col("t").min()).item()), int(group.select(pl.col("t").max()).item()))
                    )
                    - set(group.select("t").to_series())
                )
            else:
                t_missing = list(set(range(t_len)) - set(group.select("t").to_series()))
            if len(t_missing):
                missing.append(
                    pl.DataFrame(
                        {
                            "t": t_missing,
                            "cell": track_id,
                            "x": np.interp(t_missing, *group.select("t", "x").to_numpy().T),
                            "y": np.interp(t_missing, *group.select("t", "y").to_numpy().T),
                            "area": 0,
                            "track_id": track_id,
                        },
                        schema=tracks.schema,
                    )
                )
        if len(missing):
            return pl.concat(missing)
        else:
            return pl.DataFrame(schema=tracks.schema)
    else:
        for cell in tracks["label"].unique():
            h = tracks.query("label==@cell", local_dict=dict(cell=cell))
            if t_len is None:
                t_missing = list(set(range(int(h["t"].min()), int(h["t"].max()))) - set(h["t"]))
            else:
                t_missing = list(set(range(t_len)) - set(h["t"]))
            g = pandas.DataFrame(np.full((len(t_missing), tracks.shape[1]), np.nan), columns=tracks.columns)
            g["t"] = t_missing
            g["t_stamp"] = t_missing
            g["x"] = np.interp(t_missing, h["t"], h["x"])
            g["y"] = np.interp(t_missing, h["t"], h["y"])
            g["label"] = cell
            missing.append(g)
        return pandas.concat(missing, ignore_index=True)


def substitute_missing(
    tracks: pl.DataFrame | pandas.DataFrame, missing: pl.DataFrame | pandas.DataFrame, distance: int = 1
) -> pl.DataFrame | pandas.DataFrame:
    """relabel rows in tracks if they overlap with a row in missing"""
    if isinstance(tracks, pl.DataFrame):
        missing = (
            tracks.join_where(
                missing,
                (pl.col("t") == pl.col("t_r"))
                & ((pl.col("x") - pl.col("x_r")) ** 2 + (pl.col("y") - pl.col("y_r")) ** 2 < distance**2),
                suffix="_r",
            )
            .select(
                pl.all()
                .bottom_k_by((pl.col("x") - pl.col("x_r")) ** 2 + (pl.col("y") - pl.col("y_r")) ** 2, 1)
                .over("t", "cell", mapping_strategy="explode")
            )
            .select("t", "cell", "x", "y", "area", track_id="track_id_r")
        )
        return pl.concat((tracks.join(missing, ("t", "cell"), "anti"), missing)).sort("t", "cell")
    else:
        for _, row in missing.iterrows():
            a = tracks.query(
                "t==@t & (x-@x)**2 + (y-@y)**2 < @d",
                local_dict=dict(t=row["t"], x=row["x"], y=row["y"], d=distance**2),
            ).copy()
            a["label"] = row["label"]
            if len(a) == 1:
                tracks.loc[a.index[0], "label"] = row["label"]
            elif len(a) > 1:
                idx = ((a[["x", "y"]] - row[["x", "y"]].tolist()) ** 2).sum(1).idxmin()
                tracks.loc[idx, "label"] = row["label"]
        return tracks


def filter_kwargs(function: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    args = getfullargspec(function)
    return {key: value for key, value in kwargs.items() if key in args.args + args.kwonlyargs}


def get_xy(im: ArrayLike) -> tuple[float, float]:
    im = np.asarray(im)
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    s = np.nansum(im)
    return np.nansum(im * y) / s, np.nansum(im * x) / s  # type: ignore


def connect_nuclei_with_cells(nuclei: ArrayLike, cells: ArrayLike) -> np.ndarray:
    nuclei = np.asarray(nuclei)
    cells = np.asarray(cells)
    i_nuclei = np.array([i for i in np.unique(nuclei) if i > 0])
    i_cells = np.array([i for i in np.unique(cells) if i > 0])
    if len(i_nuclei) == 0 | len(i_cells) == 0:
        return np.zeros_like(cells)
    j = (nuclei.flatten()) > 0 | (cells.flatten() > 0)
    nuclei_flat = nuclei.flatten()[j]
    cells_flat = cells.flatten()[j]
    jaccard = cdist(
        np.vstack([nuclei_flat == i for i in i_nuclei]).astype(int),
        np.vstack([cells_flat == i for i in i_cells]).astype(int),
        "jaccard",
    )
    idx = np.where(jaccard < 0.95)
    nuclei_lbl, cell_lbl = np.vstack((i_nuclei[idx[0]], i_cells[idx[1]]))

    d = {}  # find groups of overlapping nuclei and cells
    for n in np.unique(nuclei_lbl):
        if not any([n in k for k in d.keys()]):
            visited_nuclei = set()
            visited_cells = set()
            nuclei_set = {n}
            cell_set = set()
            while len(nuclei_set - visited_nuclei) or len(cell_set - visited_cells):
                for i in nuclei_set - visited_nuclei:
                    cell_set |= set(cell_lbl[nuclei_lbl == i])
                visited_nuclei |= nuclei_set

                for i in cell_set - visited_cells:
                    nuclei_set |= set(nuclei_lbl[cell_lbl == i])
                visited_cells |= cell_set
            d[tuple(nuclei_set)] = tuple(cell_set)

    cells_new = np.zeros_like(cells)
    x, y = np.meshgrid(range(cells.shape[1]), range(cells.shape[0]))
    for n, c in d.items():
        if len(n) == 1:
            cells_new[np.isin(cells, c)] = n
        else:
            mask = np.zeros(cells.shape, bool)
            mask[np.isin(cells, c)] = True
            mask[np.isin(nuclei, n)] = True
            centers = [get_xy(nuclei == i) for i in n]
            dist = np.min([(x - i[1]) ** 2 + (y - i[0]) ** 2 for i in centers], 0)
            markers = np.zeros_like(cells)
            for i, n_c in zip(centers, n):
                markers[int(round(i[0])), int(round(i[1]))] = n_c
            cells_new[mask] = watershed(dist, markers=markers, mask=mask)[mask]
    cells_new[nuclei > 0] = nuclei[nuclei > 0]
    cells_new[np.isin(cells_new, np.setdiff1d(cells_new, nuclei))] = 0
    return cells_new


def trackmate_fiji(
    file_in: Path | str,
    file_out: Path | str,
    fiji_path: Path | str = None,
    channel: int = 0,
    **kwargs: dict[str, [str, int, float, bool]],  # type: ignore
) -> None:
    if fiji_path is not None:
        fiji_path = Path(fiji_path)
    if fiji_path is not None and fiji_path.exists():
        ij = imagej.init(str(fiji_path))
    else:
        ij = imagej.init("sc.fiji:fiji")
    settings = dict(
        file_in=str(file_in),
        file_out=str(file_out),
        TARGET_CHANNEL=1 + channel,
        MIN_AREA=20,
        MAX_FRAME_GAP=2,
        ALTERNATIVE_LINKING_COST_FACTOR=1.05,
        LINKING_MAX_DISTANCE=15.0,
        GAP_CLOSING_MAX_DISTANCE=15.0,
        SPLITTING_MAX_DISTANCE=15.0,
        ALLOW_GAP_CLOSING=True,
        ALLOW_TRACK_SPLITTING=False,
        ALLOW_TRACK_MERGING=False,
        MERGING_MAX_DISTANCE=15.0,
        CUTOFF_PERCENTILE=0.9,
    )
    settings.update(
        {key.upper(): value for key, value in kwargs.items() if key in settings}  # type: ignore
    )
    with open(Path(__file__).parent / "trackmate.jy") as f:
        ij.py.run_script("py", f.read(), settings)
    ij.dispose()


def lap_track(
    max_file: Path | str,
    tif_file: Path | str,
    tiff_out: Path | str,
    table_out: Path | str = None,
    min_frames: int = None,
    remove_borders: bool = False,
    nucleoli_kwargs: dict[str, Any] = None,
    min_area: int = None,
    **kwargs: dict[str, str],
) -> None:
    """use laptrack to make sure cells have the same label in all frames, relabel even if there's just one frame,
    to make sure that cell numbers are consecutive
    also remove cells/nuclei touching the border of the image if needed (only if not a time-lapse)
    """
    with ExitStack() as stack:  # noqa
        mask = stack.enter_context(Imread(tif_file, axes="ctyx"))  # noqa
        y, x = np.meshgrid(np.arange(mask.shape["y"]), np.arange(mask.shape["x"]), indexing="ij")
        cells = []
        for t in trange(mask.shape["t"], desc="generating cell mask properties", leave=False):
            cells_in_mask = np.unique(mask[-1, t])  # type: ignore
            cells_in_mask = cells_in_mask[np.isfinite(cells_in_mask) & (cells_in_mask > 0)]
            xc = []
            yc = []
            area = []
            c = []
            for cell in cells_in_mask:
                mask_t = mask[-1, t] == cell  # type: ignore
                sum_mask = np.sum(mask_t)
                if min_area is None or sum_mask >= min_area:
                    xc.append(np.sum(mask_t * x) / sum_mask)  # type: ignore
                    yc.append(np.sum(mask_t * y) / sum_mask)  # type: ignore
                    area.append(sum_mask)
                    c.append(cell)

            cells.append(
                pl.DataFrame(
                    {"t": t, "cell": c, "x": xc, "y": yc, "area": area},
                    schema={"t": pl.UInt32, "cell": pl.UInt32, "x": pl.Float64, "y": pl.Float64, "area": pl.UInt64},
                )
            )
        cells = pl.concat(cells)

        if mask.shape["t"] > 1:

            def metric(a, b):
                ta, na, xa, ya = int(a[0]), int(a[1]), float(a[2]), float(a[3])
                tb, nb, xb, yb = int(b[0]), int(b[1]), float(b[2]), float(b[3])
                dt = np.abs(ta - tb)
                cell_a = mask[-1, ta] == na  # type: ignore
                cell_b = mask[-1, tb] == nb  # type: ignore
                return (1 - np.sum(cell_a & cell_b) / np.sum(cell_a | cell_b)) * np.sqrt(  # type: ignore
                    dt * ((xa - xb) ** 2 + (ya - yb) ** 2)
                )

            lt_kwargs = filter_kwargs(LapTrack, {key.lower(): value for key, value in kwargs.items()}) | {
                "metric": metric,
                "gap_closing_metric": metric,
                "splitting_metric": metric,
                "merging_metric": metric,
                "parallel_backend": ParallelBackend.ray,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                lt = LapTrack(**lt_kwargs)
            track_df, _split_df, _merge_df = lt.predict_dataframe(
                cells.to_pandas(), coordinate_cols=["t", "cell", "x", "y"], frame_col="t", only_coordinate_cols=False
            )
            tracks = pl.from_pandas(track_df.reset_index(drop=True), schema_overrides=cells.schema).select(
                cells.columns, track_id=pl.col("track_id") + 1
            )
            missing = interpolate_missing(tracks)
            tracks = substitute_missing(tracks, missing)
            tracks = sort_labels(tracks)
            missing = interpolate_missing(tracks)
            tracks = pl.concat((tracks, missing)).select("t", "cell", "track_id")
        else:
            tracks = cells.select("t", "cell", track_id=pl.col("cell") - 1)
            missing = None

        if table_out:
            tracks.write_csv(table_out, separator="\t")

        if nucleoli_kwargs.get("remove") is True:  # noqa
            im = stack.enter_context(Imread(max_file, axes="ctyx"))[nucleoli_kwargs["channel"]]  # noqa
        else:
            im = None

        # relabel the labels according to the tracks and also add missing labels by interpolation
        swapper = SwapLabels(tracks, min_frames)
        dtype = "uint8" if swapper.tracks.select(pl.col("track_id").max()).item() <= 255 else "uint16"
        tif = stack.enter_context(IJTiffFile(tiff_out, pxsize=mask.pxsize_um, colormap="glasbey", dtype=dtype))
        for t in trange(mask.shape["t"], desc="reordering cells", leave=False):
            frame = swapper(np.asarray(mask[:, t]), t)  # type: ignore
            if im is not None:
                nucleoli = find_nucleoli(
                    im[t],
                    frame[-1],
                    **nucleoli_kwargs,
                )
                frame[0, nucleoli] = 0
            for c in range(frame.shape[0]):
                if missing is not None:
                    missing_t = missing.filter(pl.col("t") == t)
                    for cell in missing_t.select(pl.col("track_id").unique()).to_series().to_list():
                        time_points = get_time_points(
                            t,
                            missing.filter(pl.col("track_id") == cell).select("t").to_series().to_list(),
                        )
                        add_cell = interp_label(
                            t,
                            time_points,
                            [swapper(mask[c, i], i) for i in time_points],  # type: ignore
                            int(cell),
                            frame[c] > 0,
                        )
                        frame[c] += add_cell.astype(frame.dtype)
            if remove_borders and mask.shape["t"] == 1:
                for lbl in np.unique(
                    np.hstack(
                        (
                            frame[:, :, 0],
                            frame[:, :, -1],
                            frame[:, 0, :],
                            frame[:, -1, :],
                        )
                    )
                ):
                    frame[frame == lbl] = 0
                lbls = np.unique(frame)
                lbls = lbls[lbls > 0]
                frame_orig = frame.copy()
                for i, j in enumerate(lbls):
                    frame[frame_orig == j] = i

            for c in range(frame.shape[0]):
                # TODO: transforms
                tif.save(frame[c], c, 0, t)


def trackmate(
    max_file: Path | str,
    tif_file: Path | str,
    tiff_out: Path | str,
    table_out: Path | str = None,
    min_frames: int = None,
    remove_borders: bool = False,
    nucleoli_kwargs: dict[str, Any] = None,
    **kwargs: dict[str, str],
) -> None:
    """run trackmate to make sure cells have the same label in all frames, relabel even if there's just one frame,
    to make sure that cell numbers are consecutive
    also remove cells/nuclei touching the border of the image if needed (only if not a time-lapse)
    """
    warnings.warn("trackmate will be removed in the future, use lap_track instead", DeprecationWarning, stacklevel=2)

    with ExitStack() as stack:  # noqa
        mask = stack.enter_context(Imread(tif_file, axes="ctyx", dtype=int))  # noqa
        if mask.shape["t"] > 1:
            pat = re.compile(r"_ch1", re.IGNORECASE)
            xml_file = tif_file.with_suffix(".xml")
            trackmate_fiji(tif_file, xml_file, channel=1, **kwargs)
            try:
                tracks = trackmate_peak_import(str(xml_file), get_tracks=True)
                tracks.columns = [pat.sub("", column) for column in tracks.columns]
                tracks = tracks[["t", "t_stamp", "x", "y", "label", "median_intensity", "area"]]
                missing = interpolate_missing(tracks)
                tracks = substitute_missing(tracks, missing)
                tracks = sort_labels(tracks)
                missing = interpolate_missing(tracks)
                tracks = pandas.concat((tracks, missing), ignore_index=True)[["label", "median_intensity", "t"]]
            except FileNotFoundError:
                warnings.warn("trackmate encountered an error, segmentation will be saved, but not tracked!")
                tracks = []
                for t in range(mask.shape["t"]):
                    cells = np.unique(mask[:, t])  # type: ignore
                    cells = cells[cells > 0]
                    tracks.append(np.vstack((np.arange(1, len(cells) + 1), cells, t * np.ones(len(cells)))).T)
                tracks = pandas.DataFrame(np.vstack(tracks), columns=["label", "median_intensity", "t"])
                missing = None
        else:
            cells = np.unique(mask)
            cells = cells[cells > 0]
            tracks = pandas.DataFrame(
                np.vstack((np.arange(1, len(cells) + 1), cells)).T,
                columns=["label", "median_intensity"],
            ).assign(t=0)
            missing = None
        if table_out:
            tracks.to_csv(table_out, sep="\t", index=False)

        if nucleoli_kwargs.get("remove") is True:  # noqa
            im = stack.enter_context(Imread(max_file, axes="ctyx"))[nucleoli_kwargs["channel"]]  # noqa
        else:
            im = None

        # relabel the labels according to the tracks and also add missing labels by interpolation
        swapper = SwapLabels(tracks, min_frames)
        dtype = "uint8" if swapper.tracks["label"].max() <= 255 else "uint16"
        tif = stack.enter_context(IJTiffFile(tiff_out, pxsize=mask.pxsize_um, colormap="glasbey", dtype=dtype))
        for t in trange(mask.shape["t"], desc="reordering cells with trackmate", leave=False):
            frame = swapper(np.asarray(mask[:, t]), t)  # type: ignore
            if im is not None:
                nucleoli = find_nucleoli(
                    im[t],
                    frame[-1],
                    **nucleoli_kwargs,
                )
                frame[0, nucleoli] = 0
            for c in range(frame.shape[0]):
                if missing is not None:
                    missing_t = missing.query("t==@t", local_dict=dict(t=t))
                    for cell in missing_t["label"].unique():
                        time_points = get_time_points(
                            t,
                            missing.query("label==@cell", local_dict=dict(cell=cell))["t"].tolist(),
                        )
                        a = interp_label(
                            t,
                            time_points,
                            [swapper(mask[c, i], i) for i in time_points],  # type: ignore
                            int(cell),
                            frame[c] > 0,
                        )
                        frame[c] += a.astype(frame.dtype)
            if remove_borders and mask.shape["t"] == 1:
                for lbl in np.unique(
                    np.hstack(
                        (
                            frame[:, :, 0],
                            frame[:, :, -1],
                            frame[:, 0, :],
                            frame[:, -1, :],
                        )
                    )
                ):
                    frame[frame == lbl] = 0
                lbls = np.unique(frame)
                lbls = lbls[lbls > 0]
                frame_orig = frame.copy()
                for i, j in enumerate(lbls):
                    frame[frame_orig == j] = i

            for c in range(frame.shape[0]):
                # TODO: transforms
                tif.save(frame[c], c, 0, t)


def run_stardist(
    image: Path | str,
    tiff_out: Path | str,
    channel_cell: int,
    *,
    model_type: str = None,
    table_out: Path | str = None,
    tm_kwargs: dict[str, str] = None,
    rn_kwargs: dict[str, str | float] = None,
    cell_tracker: str = "trackmate",
) -> None:
    if model_type is None:
        model_type = "2D_versatile_fluo"
    if StarDist2D is None:
        raise ImportError("stardist is not installed, install stardist and numpy >= 1.20, < 2 using pip")

    with redirect_stdout(StringIO()):
        model = StarDist2D.from_pretrained(model_type)  # noqa
    tm_kwargs = tm_kwargs or {}
    rn_kwargs = rn_kwargs or {}

    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = Path(tempdir) / "tm.tif"

        with Imread(image, axes="ctyx") as im:  # noqa
            with IJTiffFile(tif_file, pxsize=im.pxsize_um) as tif:
                for t in tqdm(
                    range(im.shape["t"]),
                    total=im.shape["t"],
                    desc="running stardist",
                    disable=im.shape["t"] < 10,
                ):
                    tif.save(
                        model.predict_instances(normalize(im[channel_cell, t]))[0],  # type: ignore
                        0,
                        0,
                        t,
                    )

        rn_kwargs["channel"] = channel_cell
    if cell_tracker == "trackmate":
        trackmate(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)  # noqa
    else:
        lap_track(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)


class CellPoseTiff(IJTiffParallel):
    def __init__(
        self,
        model: models.CellposeModel,
        cp_kwargs: dict[str, str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.cp_kwargs = cp_kwargs or {}
        super().__init__(*args, **kwargs)

    def parallel(self, frame: tuple[ArrayLike, ...]) -> tuple[FrameInfo, ...]:
        if len(frame) == 1:
            cells = self.model.eval(
                np.stack(frame, 0),  # type: ignore
                channel_axis=0,
                channels=[[0, 0]],
                **self.cp_kwargs,
            )[0]
            return ((cells, 0, 0, 0),)
        else:
            cells = self.model.eval(
                np.stack(frame, 0),  # type: ignore
                channel_axis=0,
                channels=[[1, 0]],
                **self.cp_kwargs,
            )[0]
            nuclei = self.model.eval(
                np.stack(frame, 0),  # type: ignore
                channel_axis=0,
                channels=[[2, 0]],
                **self.cp_kwargs,
            )[0]
            cells = connect_nuclei_with_cells(nuclei, cells)
            return (cells, 0, 0, 0), (nuclei, 1, 0, 0)


def run_cellpose_cpu(
    image: Path | str,
    tiff_out: Path | str,
    channel_cell: int,
    channel_nuc: int = None,
    *,
    model_type: str = None,
    table_out: Path | str = None,
    cp_kwargs: dict[str, str] = None,
    tm_kwargs: dict[str, str] = None,
    rn_kwargs: dict[str, str | float] = None,
    cell_tracker: str = "trackmate",
) -> None:
    cp_kwargs = cp_kwargs or {}
    tm_kwargs = tm_kwargs or {}
    rn_kwargs = rn_kwargs or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = models.CellposeModel(gpu=False, model_type=model_type or "cyto3")
    cp_kwargs = filter_kwargs(model.eval, cp_kwargs)

    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = Path(tempdir) / "tm.tif"
        with Imread(image, axes="ctyx") as im:  # noqa
            with CellPoseTiff(model, cp_kwargs, tif_file, pxsize=im.pxsize_um) as tif:
                for t in tqdm(
                    range(im.shape["t"]),
                    total=im.shape["t"],
                    desc="running cellpose on the CPU",
                    disable=im.shape["t"] < 10,
                ):
                    tif.save(  # type: ignore
                        (im[channel_cell, t],) if channel_nuc is None else (im[channel_cell, t], im[channel_nuc, t]),  # type: ignore
                        0,
                        0,
                        t,
                    )
        rn_kwargs["channel"] = channel_cell if channel_nuc is None else channel_nuc
        if cell_tracker == "trackmate":
            trackmate(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)  # noqa
        else:
            lap_track(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)


def run_cellpose_cpu_serial(
    image: Path | str,
    tiff_out: Path | str,
    channel_cell: int,
    channel_nuc: int = None,
    *,
    model_type: str = None,
    table_out: Path | str = None,
    cp_kwargs: dict[str, str] = None,
    tm_kwargs: dict[str, str] = None,
    rn_kwargs: dict[str, str | float] = None,
    cell_tracker: str = "trackmate",
) -> None:
    cp_kwargs = cp_kwargs or {}
    tm_kwargs = tm_kwargs or {}
    rn_kwargs = rn_kwargs or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = models.CellposeModel(gpu=False, model_type=model_type or "cyto3")
    cp_kwargs = filter_kwargs(model.eval, cp_kwargs)

    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = Path(tempdir) / "tm.tif"
        with Imread(image, axes="ctyx") as im:  # noqa
            with IJTiffFile(tif_file, pxsize=im.pxsize_um) as tif:
                for t in tqdm(
                    range(im.shape["t"]),
                    total=im.shape["t"],
                    desc="running cellpose on the CPU",
                    disable=im.shape["t"] < 10,
                ):
                    if channel_nuc is None:
                        cells = model.eval(
                            np.stack((im[channel_cell, t],), 0),  # type: ignore
                            channel_axis=0,
                            channels=[[0, 0]],
                            **cp_kwargs,
                        )[0]
                        tif.save(cells, 0, 0, t)
                    else:
                        cells = model.eval(
                            np.stack((im[channel_cell, t], im[channel_nuc, t]), 0),  # type: ignore
                            channel_axis=0,
                            channels=[[1, 0]],
                            **cp_kwargs,
                        )[0]
                        nuclei = model.eval(
                            np.stack((im[channel_cell, t], im[channel_nuc, t]), 0),  # type: ignore
                            channel_axis=0,
                            channels=[[2, 0]],
                            **cp_kwargs,
                        )[0]
                        cells = connect_nuclei_with_cells(nuclei, cells)
                        tif.save(cells, 0, 0, t)
                        tif.save(nuclei, 1, 0, t)

        rn_kwargs["channel"] = channel_cell if channel_nuc is None else channel_nuc
        if cell_tracker == "trackmate":
            trackmate(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)  # noqa
        else:
            lap_track(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)


def run_cellpose_gpu(
    image: Path | str,
    tiff_out: Path | str,
    channel_cell: int,
    channel_nuc: int = None,
    *,
    model_type: str = None,
    table_out: Path | str = None,
    cp_kwargs: dict[str, str] = None,
    tm_kwargs: dict[str, str] = None,
    rn_kwargs: dict[str, str | float] = None,
    cell_tracker: str = "trackmate",
) -> None:
    cp_kwargs = cp_kwargs or {}
    tm_kwargs = tm_kwargs or {}
    rn_kwargs = rn_kwargs or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = models.CellposeModel(gpu=True, model_type=model_type or "cyto3")
    cp_kwargs = filter_kwargs(model.eval, cp_kwargs)
    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = Path(tempdir) / "tm.tif"
        with Imread(image, axes="ctyx") as im:  # noqa
            with IJTiffFile(tif_file, pxsize=im.pxsize_um) as tif:
                for t in tqdm(
                    range(im.shape["t"]),
                    total=im.shape["t"],
                    desc="running cellpose on a GPU",
                    disable=im.shape["t"] < 10,
                ):
                    if channel_nuc is None:
                        frame = np.stack((im[channel_cell, t],), 0)  # type: ignore
                        cells = model.eval(
                            frame,  # type: ignore
                            channel_axis=0,
                            channels=[[0, 0]],
                            **cp_kwargs,
                        )[0]
                        tif.save(cells, 0, 0, t)
                    else:
                        frame = np.stack((im[channel_cell, t], im[channel_nuc, t]), 0)  # type: ignore
                        cells = model.eval(
                            np.stack(frame, 0),  # noqa
                            channel_axis=0,
                            channels=[[1, 0]],
                            **cp_kwargs,
                        )[0]
                        nuclei = model.eval(
                            np.stack(frame, 0),  # noqa
                            channel_axis=0,
                            channels=[[2, 0]],
                            **cp_kwargs,
                        )[0]
                        cells = connect_nuclei_with_cells(nuclei, cells)
                        tif.save(cells, 0, 0, t)
                        tif.save(nuclei, 1, 0, t)
        rn_kwargs["channel"] = channel_cell if channel_nuc is None else channel_nuc
        if cell_tracker == "trackmate":
            trackmate(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)  # noqa
        else:
            lap_track(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)


@wraps(run_cellpose_cpu)
def run_cellpose(*args, **kwargs) -> None:
    if len(tensorflow.config.list_physical_devices("GPU")):
        run_cellpose_gpu(*args, **kwargs)
    else:
        run_cellpose_cpu(*args, **kwargs)


class FindCellsTiff(IJTiffParallel):
    def __init__(self, fc_kwargs: dict[str, str] = None, *args: Any, **kwargs: Any) -> None:
        self.fc_kwargs = fc_kwargs or {}
        super().__init__(*args, **kwargs)

    def parallel(self, frame: tuple[ArrayLike]) -> tuple[FrameInfo, FrameInfo]:
        cell, nucleus = findcells(*frame, **self.fc_kwargs)
        return (cell, 0, 0, 0), (nucleus, 1, 0, 0)


def run_findcells(
    image: Path | str,
    tiff_out: Path | str,
    channel_cell: int,
    channel_nuc: int = None,
    *,
    table_out: Path | str = None,
    fc_kwargs: dict[str, str] = None,
    tm_kwargs: dict[str, str] = None,
    rn_kwargs: dict[str, str | float] = None,
    cell_tracker: str = "trackmate",
) -> None:
    fc_kwargs = fc_kwargs or {}
    tm_kwargs = tm_kwargs or {}
    rn_kwargs = rn_kwargs or {}
    fc_kwargs = filter_kwargs(findcells, fc_kwargs)

    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = Path(tempdir) / "tm.tif"

        with Imread(image, axes="ctyx") as im:  # noqa
            with FindCellsTiff(fc_kwargs, tif_file, pxsize=im.pxsize_um) as tif:
                for t in tqdm(
                    range(im.shape["t"]),
                    total=im.shape["t"],
                    desc="running findcells",
                    disable=im.shape["t"] < 10,
                ):
                    assert channel_cell is not None, "channel_cell cannot be None"
                    tif.save(  # type: ignore
                        (im[channel_cell, t],) if channel_nuc is None else (im[channel_cell, t], im[channel_nuc, t]),  # type: ignore
                        0,
                        0,
                        t,
                    )

        rn_kwargs["channel"] = channel_cell if channel_nuc is None else channel_nuc
        if cell_tracker == "trackmate":
            trackmate(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)  # noqa
        else:
            lap_track(image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs)


class PreTrackTiff(IJTiffParallel):
    def __init__(self, shape_yx: tuple[int, int], radius: float, *args: Any, **kwargs: Any):
        self.shape_yx = shape_yx
        self.xv, self.yv = np.meshgrid(*[range(i) for i in shape_yx])
        self.radius = radius
        super().__init__(*args, **kwargs)

    def parallel(self, cxy: tuple[np.ndarray]) -> tuple[FrameInfo]:
        frame = np.zeros(self.shape_yx, int)
        cxy = cxy[0]
        if len(cxy):
            dist = np.round(np.min([(self.yv - i[1]) ** 2 + (self.xv - i[2]) ** 2 for i in cxy], 0)).astype(int)

            for i in cxy:
                frame[int(i[1]), int(i[2])] = i[0]  # noqa
            frame = watershed(dist, frame, mask=dist < self.radius**2)
        return ((frame, 0, 0, 0),)


def run_pre_track(
    image: Path | str, tiff_out: Path | str, pre_track: pl.DataFrame | pandas.DataFrame, radius: float
) -> None:
    if isinstance(pre_track, pandas.DataFrame):
        pre_track = pl.from_pandas(pre_track)

    dtype = "uint8" if pre_track.select(pl.col("cell").max()).item() < 255 else "uint16"
    with Imread(image) as im:  # noqa
        with PreTrackTiff(
            im.shape["yx"],  # type: ignore
            radius,
            tiff_out,
            pxsize=im.pxsize_um,
            colormap="glasbey",
            dtype=dtype,
        ) as tif:
            for t in tqdm(
                range(im.shape["t"]),
                total=im.shape["t"],
                desc="running pre track cell masking",
                disable=im.shape["t"] < 10,
            ):
                tif.save(  # type: ignore
                    (pre_track.filter(pl.col("T") == t).select("cell", "y", "x").to_numpy(),),
                    0,
                    0,
                    t,
                )


def find_nucleoli(image: ArrayLike[Any], mask: ArrayLike[int], **kwargs) -> NDArray[bool]:  # type: ignore
    def fill_mask(img: ArrayLike[Any], msk: ArrayLike[bool]) -> tuple[tuple[int, int, int, int], NDArray[Any]]:  # type: ignore
        w = np.where(msk)
        i0, j0 = np.min(w, 1)  # noqa
        i1, j1 = np.max(w, 1)  # noqa
        img = np.asarray(img[i0 : i1 + 1, j0 : j1 + 1]).copy()  # type: ignore
        msk = np.asarray(msk[i0 : i1 + 1, j0 : j1 + 1]).copy()  # type: ignore
        dist = img[msk]
        a, b = np.percentile(dist, (25, 90))
        dist = dist[(a <= dist) & (dist <= b)]
        msk[img < a] = False
        img[~msk] = np.random.choice(dist, (~msk).sum())
        return (i0, j0, i1, j1), img

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = models.CellposeModel(gpu=True, model_type=kwargs.get("model_type", "cyto3"))
    labels = [label for label in np.unique(mask) if label > 0]
    if labels:
        i, images = zip(*[fill_mask(image, mask == label) for label in np.unique(mask) if label > 0])
        kwargs["max_size_fraction"] = 1.0
        masks = model.eval(list(images), **filter_kwargs(model.eval, kwargs))[0]
        mask = np.zeros_like(mask, dtype=bool)
        for (i0, j0, i1, j1), m in zip(i, masks):
            mask[i0 : i1 + 1, j0 : j1 + 1] |= m > 0
        return mask
    else:
        return np.zeros_like(mask, bool)
