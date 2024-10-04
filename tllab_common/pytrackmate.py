import re
import xml.etree.cElementTree as et

import numpy as np
import pandas as pd


""" 
    Copyright (c) 2018, Hadrien Mary
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    
    Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    Neither the name of the read_roi nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def trackmate_peak_import(trackmate_xml_path, get_tracks=False):
    """Import detected peaks with TrackMate Fiji plugin.

    Parameters
    ----------
    trackmate_xml_path : str
        TrackMate XML file path.
    get_tracks : boolean
        Add tracks to label
    """

    root = et.fromstring(open(trackmate_xml_path).read())

    object_labels = {'FRAME': 't_stamp',
                     'POSITION_T': 't',
                     'POSITION_X': 'x',
                     'POSITION_Y': 'y',
                     'POSITION_Z': 'z',
                     'ESTIMATED_DIAMETER': 'w',
                     'QUALITY': 'q',
                     'ID': 'spot_id',
                     'MEAN_INTENSITY': 'mean_intensity',
                     'MEDIAN_INTENSITY': 'median_intensity',
                     'MIN_INTENSITY': 'min_intensity',
                     'MAX_INTENSITY': 'max_intensity',
                     'TOTAL_INTENSITY': 'total_intensity',
                     'STANDARD_DEVIATION': 'std_intensity',
                     'CONTRAST': 'contrast',
                     'SNR': 'snr'}

    features = root.find('Model').find('FeatureDeclarations').find('SpotFeatures')
    features = [c.get('feature') for c in list(features)] + ['ID']

    spots = root.find('Model').find('AllSpots')
    objects = []
    for frame in spots.findall('SpotsInFrame'):
        for spot in frame.findall('Spot'):
            single_object = []
            for label in features:
                single_object.append(spot.get(label))
            objects.append(single_object)

    trajs = pd.DataFrame(objects, columns=features)
    trajs = trajs.astype(float)

    # Apply initial filtering
    initial_filter = root.find("Settings").find("InitialSpotFilter")

    trajs = filter_spots(trajs,
                         name=initial_filter.get('feature'),
                         value=float(initial_filter.get('value')),
                         isabove=True if initial_filter.get('isabove') == 'true' else False)

    # Apply filters
    spot_filters = root.find("Settings").find("SpotFilterCollection")

    for spot_filter in spot_filters.findall('Filter'):

        trajs = filter_spots(trajs,
                             name=spot_filter.get('feature'),
                             value=float(spot_filter.get('value')),
                             isabove=True if spot_filter.get('isabove') == 'true' else False)

    trajs.columns = [re.sub(r'_CH\d', '', column) for column in trajs.columns]
    trajs.rename(columns=object_labels, inplace=True)
    trajs.columns = [column.lower() for column in trajs.columns]
    trajs['label'] = np.arange(trajs.shape[0])

    # Get tracks
    if get_tracks:
        filtered_track_ids = [int(track.get('TRACK_ID')) for track in root.find('Model').find('FilteredTracks').findall('TrackID')]

        label_id = 0
        trajs['label'] = np.nan

        tracks = root.find('Model').find('AllTracks')
        for track in tracks.findall('Track'):

            track_id = int(track.get("TRACK_ID"))
            if track_id in filtered_track_ids:

                spot_ids = [(edge.get('SPOT_SOURCE_ID'), edge.get('SPOT_TARGET_ID'), edge.get('EDGE_TIME')) for edge in track.findall('Edge')]
                spot_ids = np.array(spot_ids).astype('float')[:, :2]
                spot_ids = set(spot_ids.flatten())

                trajs.loc[trajs["spot_id"].isin(spot_ids), "label"] = label_id
                label_id += 1

        # Label remaining columns
        single_track = trajs.loc[trajs["label"].isnull()]
        trajs.loc[trajs["label"].isnull(), "label"] = label_id + np.arange(0, len(single_track))

    return trajs


def filter_spots(spots, name, value, isabove):
    if isabove:
        spots = spots[spots[name] > value]
    else:
        spots = spots[spots[name] < value]

    return spots
