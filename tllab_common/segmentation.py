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

    logger = tensorflow.get_logger()
    logger.setLevel(logging.ERROR)

import numpy as np
import pandas
from cellpose import models
from csbdeep.utils import normalize
from ndbioimage import Imread
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed
from stardist.models import StarDist2D  # noqa
from tiffwrite import FrameInfo, IJTiffFile, IJTiffParallel
from tqdm.auto import tqdm, trange

from .findcells import findcells
from .pytrackmate import trackmate_peak_import

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
    def __init__(self, tracks: pandas.DataFrame, min_frames: int = None) -> None:
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
            tracks["label"] = tracks.apply(lambda s: d[s["label"]], 1)
        self.tracks = tracks

    def __call__(
        self, im: Imread, frame_in: np.ndarray, c: int, z: int, t: int
    ) -> np.ndarray:
        frame_out = np.zeros_like(frame_in)
        for i, j in self.get_mapping(t).items():
            frame_out[frame_in == i] = j
        return frame_out

    def get_mapping(self, t: int) -> dict[int, int]:
        return {
            int(i): int(j)
            for i, j in self.tracks.query("t == @t", local_dict=dict(t=t))[
                ["median_intensity", "label"]
            ].to_numpy()
            if np.isfinite(i)
        }

    def get_labels(self) -> np.ndarray:
        return self.tracks["label"].unique()


def sort_labels(tracks: pandas.DataFrame) -> pandas.DataFrame:
    """make labels consistent across different runs"""
    relabel_dict = {
        int(key): value
        for value, key in enumerate(
            tracks.groupby("label").aggregate("mean").sort_values("area").index, 1
        )
    }
    return tracks.groupby("label").apply(
        lambda x: x.assign(label=relabel_dict[x["label"].mean()])
    )


def get_time_points(t: int, missing: Sequence[int]) -> tuple[int, int]:
    t_a = t - 1
    while t_a in missing:
        t_a -= 1
    t_b = t + 1
    while t_b in missing:
        t_b += 1
    return t_a, t_b


def interpolate_missing(
    tracks: pandas.DataFrame, t_len: int = None
) -> pandas.DataFrame:
    """interpolate the position of the cell in missing frames"""
    missing = []
    for cell in tracks["label"].unique():
        h = tracks.query("label==@cell", local_dict=dict(cell=cell))
        if t_len is None:
            t_missing = list(
                set(range(int(h["t"].min()), int(h["t"].max()))) - set(h["t"])
            )
        else:
            t_missing = list(set(range(t_len)) - set(h["t"]))
        g = pandas.DataFrame(
            np.full((len(t_missing), tracks.shape[1]), np.nan), columns=tracks.columns
        )
        g["t"] = t_missing
        g["t_stamp"] = t_missing
        g["x"] = np.interp(t_missing, h["t"], h["x"])
        g["y"] = np.interp(t_missing, h["t"], h["y"])
        g["label"] = cell
        missing.append(g)
    return pandas.concat(missing, ignore_index=True)


def substitute_missing(
    tracks: pandas.DataFrame, missing: pandas.DataFrame, distance: int = 1
) -> pandas.DataFrame:
    """relabel rows in tracks if they overlap with a row in missing"""
    for _, row in missing.iterrows():
        a = tracks.query(
            "t==@t & (x-@x)**2 + (y-@y)**2 < @distance",
            local_dict=dict(t=row["t"], x=row["x"], y=row["y"], distance=distance),
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
    return {
        key: value
        for key, value in kwargs.items()
        if key in args.args + args.kwonlyargs
    }


def get_xy(im: ArrayLike) -> tuple[float, float]:
    im = np.asarray(im)
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    s = np.sum(im)
    return np.sum(im * y) / s, np.sum(im * x) / s


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

    with ExitStack() as stack:
        mask = stack.enter_context(Imread(tif_file, axes="ctyx", dtype=int))
        if mask.shape["t"] > 1:
            pat = re.compile(r"_ch1", re.IGNORECASE)
            xml_file = tif_file.with_suffix(".xml")
            trackmate_fiji(tif_file, xml_file, channel=1, **kwargs)
            tracks = trackmate_peak_import(str(xml_file), get_tracks=True)
            tracks.columns = [pat.sub("", column) for column in tracks.columns]
            tracks = tracks[
                ["t", "t_stamp", "x", "y", "label", "median_intensity", "area"]
            ]
            missing = interpolate_missing(tracks)
            tracks = substitute_missing(tracks, missing)
            tracks = sort_labels(tracks)
            missing = interpolate_missing(tracks)
            tracks = pandas.concat((tracks, missing), ignore_index=True)[
                ["label", "median_intensity", "t"]
            ]
        else:
            cells = np.unique(mask)
            tracks = pandas.DataFrame(
                np.vstack((np.arange(1, len(cells)), cells[cells > 0])).T,
                columns=["label", "median_intensity"],
            ).assign(t=0)
            missing = None
        if table_out:
            tracks.to_csv(table_out, sep="\t", index=False)

        if nucleoli_kwargs.get("remove") is True:
            im = stack.enter_context(Imread(max_file, axes="ctyx"))[
                nucleoli_kwargs["channel"]
            ]
        else:
            im = None

        # relabel the labels according to the tracks and also add missing labels by interpolation
        mask.frame_decorator = SwapLabels(tracks, min_frames)
        dtype = (
            "uint8" if mask.frame_decorator.tracks["label"].max() <= 255 else "uint16"
        )
        tif = stack.enter_context(
            IJTiffFile(tiff_out, pxsize=mask.pxsize_um, colormap="glasbey", dtype=dtype)
        )
        for t in trange(
            mask.shape["t"], desc="reordering cells with trackmate", leave=False
        ):
            frame = np.asarray(mask[:, t])
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
                            missing.query("label==@cell", local_dict=dict(cell=cell))[
                                "t"
                            ].tolist(),
                        )
                        a = interp_label(
                            t,
                            time_points,
                            [mask[c, i] for i in time_points],  # type: ignore
                            int(cell),
                            frame[c] > 0,
                        )
                        frame[c] += a
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
) -> None:
    if model_type is None:
        model_type = "2D_versatile_fluo"
    with redirect_stdout(StringIO()):
        model = StarDist2D.from_pretrained(model_type)
    tm_kwargs = tm_kwargs or {}
    rn_kwargs = rn_kwargs or {}

    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = Path(tempdir) / "tm.tif"

        with Imread(image, axes="ctyx") as im:
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
        trackmate(
            image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs
        )


class CellPoseTiff(IJTiffParallel):
    def __init__(
        self,
        model: models.Cellpose,
        cp_kwargs: dict[str, str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.cp_kwargs = cp_kwargs or {}
        super().__init__(*args, **kwargs)

    def parallel(self, frame: tuple[ArrayLike]) -> tuple[FrameInfo, ...]:
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
) -> None:
    cp_kwargs = cp_kwargs or {}
    tm_kwargs = tm_kwargs or {}
    rn_kwargs = rn_kwargs or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = models.Cellpose(gpu=False, model_type=model_type)
    cp_kwargs = filter_kwargs(model.eval, cp_kwargs)

    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = Path(tempdir) / "tm.tif"
        with Imread(image, axes="ctyx") as im:
            with CellPoseTiff(model, cp_kwargs, tif_file, pxsize=im.pxsize_um) as tif:
                for t in tqdm(
                    range(im.shape["t"]),
                    total=im.shape["t"],
                    desc="running cellpose",
                    disable=im.shape["t"] < 10,
                ):
                    tif.save(  # type: ignore
                        (im[channel_cell, t],)
                        if channel_nuc is None
                        else (im[channel_cell, t], im[channel_nuc, t]),
                        0,
                        0,
                        t,
                    )
        rn_kwargs["channel"] = channel_cell if channel_nuc is None else channel_nuc
        trackmate(
            image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs
        )


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
) -> None:
    cp_kwargs = cp_kwargs or {}
    tm_kwargs = tm_kwargs or {}
    rn_kwargs = rn_kwargs or {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = models.Cellpose(gpu=True, model_type=model_type)
    cp_kwargs = filter_kwargs(model.eval, cp_kwargs)
    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = Path(tempdir) / "tm.tif"
        with Imread(image, axes="ctyx") as im:
            with IJTiffFile(tif_file, pxsize=im.pxsize_um) as tif:
                for t in tqdm(
                    range(im.shape["t"]),
                    total=im.shape["t"],
                    desc="running cellpose",
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
        trackmate(
            image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs
        )


@wraps(run_cellpose_cpu)
def run_cellpose(*args, **kwargs) -> None:
    if len(tensorflow.config.list_physical_devices("GPU")):
        run_cellpose_gpu(*args, **kwargs)
    else:
        run_cellpose_cpu(*args, **kwargs)


class FindCellsTiff(IJTiffParallel):
    def __init__(
        self, fc_kwargs: dict[str, str] = None, *args: Any, **kwargs: Any
    ) -> None:
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
) -> None:
    fc_kwargs = fc_kwargs or {}
    tm_kwargs = tm_kwargs or {}
    rn_kwargs = rn_kwargs or {}
    fc_kwargs = filter_kwargs(findcells, fc_kwargs)

    with tempfile.TemporaryDirectory() as tempdir:
        tif_file = Path(tempdir) / "tm.tif"

        with Imread(image, axes="ctyx") as im:
            with FindCellsTiff(fc_kwargs, tif_file, pxsize=im.pxsize_um) as tif:
                for t in tqdm(
                    range(im.shape["t"]),
                    total=im.shape["t"],
                    desc="running findcells",
                    disable=im.shape["t"] < 10,
                ):
                    assert channel_cell is not None, "channel_cell cannot be None"
                    tif.save(  # type: ignore
                        (im[channel_cell, t],)
                        if channel_nuc is None
                        else (im[channel_cell, t], im[channel_nuc, t]),
                        0,
                        0,
                        t,
                    )

        rn_kwargs["channel"] = channel_cell if channel_nuc is None else channel_nuc
        trackmate(
            image, tif_file, tiff_out, table_out, nucleoli_kwargs=rn_kwargs, **tm_kwargs
        )


class PreTrackTiff(IJTiffParallel):
    def __init__(
        self, shape_yx: tuple[int, int], radius: float, *args: Any, **kwargs: Any
    ):
        self.shape_yx = shape_yx
        self.xv, self.yv = np.meshgrid(*[range(i) for i in shape_yx])
        self.radius = radius
        super().__init__(*args, **kwargs)

    def parallel(self, cxy: tuple[np.ndarray]) -> tuple[FrameInfo]:
        frame = np.zeros(self.shape_yx, int)
        cxy = cxy[0]
        if len(cxy):
            dist = np.round(
                np.min([(self.yv - i[1]) ** 2 + (self.xv - i[2]) ** 2 for i in cxy], 0)
            ).astype(int)

            for i in cxy:
                frame[int(i[1]), int(i[2])] = i[0]  # noqa
            frame = watershed(dist, frame, mask=dist < self.radius**2)
        return ((frame, 0, 0, 0),)


def run_pre_track(
    image: Path | str, tiff_out: Path | str, pre_track: pandas.DataFrame, radius: float
) -> None:
    dtype = "uint8" if pre_track["cell"].max() < 255 else "uint16"
    with Imread(image) as im:
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
                    (
                        pre_track.query("T == @t", local_dict=dict(t=t))[
                            ["cell", "y", "x"]
                        ].to_numpy(),
                    ),
                    0,
                    0,
                    t,
                )


def find_nucleoli(
    image: ArrayLike[Any], mask: ArrayLike[int], **kwargs
) -> NDArray[bool]:
    def fill_mask(img: ArrayLike[Any], msk: ArrayLike[bool]) -> NDArray[Any]:
        img = np.asarray(img).copy()
        msk = np.asarray(msk).copy()
        dist = img[msk].flatten()
        if not len(dist):
            return img
        a, b = np.percentile(dist, (25, 90))
        dist = dist[(a <= dist) & (dist <= b)]
        if not len(dist):
            return img
        msk[img < a] = False
        img[~msk] = np.random.choice(dist, np.sum(~msk))
        return img

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = models.Cellpose(
            gpu=len(tensorflow.config.list_physical_devices("GPU")) > 0,
            model_type="cyto3",
        )
    labels = [label for label in np.unique(mask) if label > 0]
    if labels:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                np.max(
                    model.eval(
                        [
                            fill_mask(image, mask == label)  # type: ignore
                            for label in labels
                        ],
                        **filter_kwargs(model.eval, kwargs),
                    )[0],
                    0,
                )
                > 0
            )
    else:
        return np.zeros(mask.shape, bool)
