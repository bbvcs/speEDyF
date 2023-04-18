import os
import datetime
import math
import json


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyedflib

from .utils.custom_print import print
from .utils import constants
#from .utils.resampling import all_header_indicated_sample_rates

"""
def edf_collate_OLD(root, out):

    edf_files = []
    bad_files = []

    # collect edf files under root directory
    for path, subdirs, files in os.walk(root):
        for file in files:
            if file.split(".")[-1].lower() in ["edf", "edf+", "bdf", "bdf+"]:
                edf_files.append(os.path.join(path, file))

    # read headers and store in dict of filepath -> header
    edf_headers = {}
    for file in edf_files:
        try:
            edf_headers[file] = pyedflib.highlevel.read_edf_header(file, read_annotations=False)
        except OSError as e:
            print(f"\tCould not read {file} ({e})", enabled=constants.VERBOSE)
            bad_files.append(file)

    for file in bad_files:
        edf_files.remove(file)

    # sort edf_files by start date
    edf_files = sorted(edf_files, key=lambda x: edf_headers[x]["startdate"])

    # rearrange edf_headers now that we've sorted files
    edf_headers = {file:edf_headers[file] for file in edf_files}

    # use headers/signalheaders to pick a sample rate
    edf_sample_rate = all_header_indicated_sample_rates(edf_headers)[0]
    # TODO resampling - it HAS TO BE DONE BY THIS POINT!

    # using headers, construct matrix of which channels present in which files
    edf_channels_dict = {file: header["channels"] for file, header in edf_headers.items()}
    edf_channels_superset = sorted(set.union(*map(set, edf_channels_dict.values())))
    edf_channels_ndarray = np.zeros(shape=(len(edf_channels_superset), len(edf_files)), dtype=np.int8)
    for i, file in enumerate(edf_files):
        for edf_channel_intervals, channel_label in enumerate(edf_channels_superset):
            # if the channel is present in this file, set corresponding matrix pos to 1
            if channel_label in edf_headers[file]["channels"]:
                edf_channels_ndarray[edf_channel_intervals, i] = 1
    edf_channels_mtx = pd.DataFrame(edf_channels_ndarray, index=edf_channels_superset, columns=edf_files, dtype=np.int8)

    # for each channel, get the number of files it appears in
    edf_channels_counts = {channel:np.sum(edf_channels_mtx.loc[channel]) for channel in edf_channels_superset}
    min_channel_count = len(edf_files) * 0.75
    excluded_channels = [channel for channel, count in edf_channels_counts.items() if count < min_channel_count]
    if len(excluded_channels) > 0:
        print("Channels {} will be excluded, as they are in less than 75% of {} files (counts {}, 75% thresh: {})".format(
                excluded_channels, len(edf_files),
                [edf_channels_counts[channel] for channel in excluded_channels], min_channel_count),
            enabled=constants.VERBOSE)
    # TODO exclude (set to NaN in collation matrix)

    # using headers, collect start/end times for each file
    edf_intervals = {}
    for file in edf_files:
        start = edf_headers[file]["startdate"]
        end = start + datetime.timedelta(seconds=edf_headers[file]["Duration"])

        edf_intervals[file] = (start, end)

    # get list of tuples containing (file, (start, end))
    zipped_intervals = list(zip(edf_intervals.keys(), edf_intervals.values()))

    # get every unique pair of intervals
    #combinations = list(itertools.combinations(zipped_intervals, 2))
    file_combinations = list(itertools.combinations(edf_files, 2))

    # set up the resolution matrix; for any pair of overlapping files, add details of the change we have made
    # this list will contain ResolutionMatrixEntry objects
    # this will be used to inform user of decisions made regarding overlaps
    resolution_mtx_entries = []

    # define time intervals for each channel; specific datetimes when, each channel starts/stops, adjusted for overlaps
    # this will be used to determine if files overlap, and is continuously adjusted as amendments to files are made
    edf_channel_intervals = {}
    for file, channels in edf_channels_dict.items():
        edf_channel_intervals[file] = {}
        for channel_label in channels:
            # for each channel in each file, list of tuples representing read intervals
            # initialise with (0, (length of recording)) for each file/channel (i.e, read all)
            # then if we omit sections of channels due to overlap, update the channel's list
            #edf_channel_intervals[file][channel] = [(0, edf_headers[file]["Duration"]*edf_sample_rate)]

            # for each channel in each file, list of tuples representing start/end times of channels
            # initialise to read all of a channel (the start/end of file it is in in edf_intervals
            # then if we omit any data from channel due to overlap, update the intervals for the channel
            edf_channel_intervals[file][channel_label] = [edf_intervals[file]]
    # BOTH resolution_mtx_entries and edf_channel_intervals are updated at the same time and should reflect eachother

    overlapping_pair_count = 0
    for file_pair in file_combinations:

        file_a = file_pair[0]
        file_b = file_pair[1]

        file_a_header = edf_headers[file_a]
        file_b_header = edf_headers[file_b]

        channel_check_results = check_channel_overlap(file_a_header, file_b_header)
        if len(channel_check_results) != 0:  # if there are common channels

            common_channels = channel_check_results[0]

            for channel_label in common_channels:

                # what are the intervals (start/end times) for this channel in each file?
                channel_file_a_intervals = edf_channel_intervals[file_a][channel_label]
                channel_file_b_intervals = edf_channel_intervals[file_b][channel_label]

                # check each pair of intervals across the 2 files
                # (note, in most circumstances, there will only be 1 interval in each list,
                # however, if we happen to omit some data in the middle of a file, it will have two intervals
                for channel_file_a_interval in channel_file_a_intervals:
                    for channel_file_b_interval in channel_file_b_intervals:

                        overlap_type, overlap_period = check_time_overlap(channel_file_a_interval[0], channel_file_a_interval[1],
                                                                          channel_file_b_interval[0], channel_file_b_interval[1])

                        if overlap_type != OverlapType.NO_OVERLAP:
                            print(str(overlap_type), enabled=constants.VERBOSE)

                            resolve_overlap(overlapping_pair_count, resolution_mtx_entries, edf_channel_intervals, edf_sample_rate,
                                            channel_label, channel_file_a_interval, channel_file_b_interval,
                                            file_a, file_a_header,
                                            file_b, file_b_header,
                                            overlap_type, overlap_period)

        overlapping_pair_count += 1  # TODO this needs to be changed, will go up on non-overlapping pairs

    # overlapping_pair_count = 0
    # for pair in combinations:
    #     file_a = pair[0]
    #     file_b = pair[1]
    #
    #     file_a_start = file_a[1][0]
    #     file_a_end = file_a[1][1]
    #     file_b_start = file_b[1][0]
    #     file_b_end = file_b[1][1]
    #
    #     file_a_path = file_a[0]
    #     file_b_path = file_b[0]
    #
    #     file_a_header = edf_headers[file_a_path]
    #     file_b_header = edf_headers[file_b_path]
    #
    #     # firstly, do they overlap in time?
    #     overlap_type, overlap_period = check_time_overlap(file_a_start, file_a_end, file_b_start, file_b_end)
    #
    #     if overlap_type != OverlapType.NO_OVERLAP:
    #         print(str(overlap_type), enabled=constants.VERBOSE)
    #
    #         # OK, but do they have the same channels?
    #         #   > if not, no problem; there isn't any overlap.
    #         #     - this can be the case with concurrent EDF files representing different groups of channels
    #         #   > if so, potentially big problem - but is the data in these overlapping channels the same?
    #         #     - if so, less of a problem. depending on overlap_type, we can resolve in different ways
    #         #        > essentially, leaving the data in would be OK; same data at same time, even if overwrite no change
    #         #     - if not, how can we know which to keep? some potential solutions:
    #         #        - allow visual inspection of overlapping data in both files, let user decide which to keep
    #         #           - slow, problematic if overlap period long
    #         #           - letting the user do this manually with EDF visualisation
    #         #           - this was approach of EDFProcessor, not going to try that this time.
    #         #        - try to determine which channel has more "information"
    #         #           - e.g maybe one channel is essentially 0, or compelte noise, whereas anbother is valid EDF
    #         #           * might give this a go, but not sure how to approach
    #         #           - problem is, what if both data seems valid?
    #         #        * omit the data that is overlapping in *both* files
    #         #           - essentially, if both data looks valid, how can we know which is true? even with inspection
    #         #           - in any case, the overlapping period may be so small (couple sec) that this is most sensible
    #
    #         channel_check_results = check_channel_overlap(file_a_header, file_b_header)
    #         if len(channel_check_results) != 0:  # if there are common channels
    #
    #
    #
    #             resolve_overlap(overlapping_pair_count, resolution_mtx_entries, edf_channel_intervals, edf_sample_rate, channel_check_results,
    #                             file_a_path, file_a_header,
    #                             file_b_path, file_b_header,
    #                             overlap_type, overlap_period)
    #
    #             overlapping_pair_count += 1



    resolution_mtx = pd.DataFrame([entry.as_dict() for entry in resolution_mtx_entries])
    resolution_mtx.to_csv(os.path.join(out, "resolution_matrix.csv"))

    # channels_intervals = []
    # for file, channels in edf_channel_intervals.items():
    #     for channel, intervals in channels.items():
    #         channels_intervals.append(intervals[0])
    # fig, ax = interval_plot(channels_intervals)
    # fig.show()


    return
"""

def edf_collate(root, out):
    """ Produce a matrix representation of a chronologically ordered collection of EDF files.

    Given a root directory, find all EDF files present in the directory and any subdirectories.
    Using their headers, order them chronologically, and produce a "Logical Collation" matrix of these files.

    A "Physical Collation" (abbr. physicol) of these EDF files would be a large EDF file on disk, containing the data of all the EDF files
    under the root directory, concatenated with gaps in recording time accounted for.

    The "Logical Collation" (abbr. logicol) is a matrix representation of the physicol, where files are chronologically
    ordered, and where each channel in a file has its own row, with start/end idx of that channel within the
    hypothetical physical collation. This can be then used by other programs in the pipeline, such as by edf_segment to
    determine how to break up the data into segments, and which files to extract data for a segment from.

    NOTE:
    - This method produces only the logicol, which could later be used to produce the physicol if necessary.
    - This method *does not* check for any overlap between EDF files - there may be overlapping data in the logicol.
        - see edf_overlaps

    :param root: a directory containing EDF files and/or subdirectories with EDF files etc.
    :param out: a directory to save the collation outputs.
    :return: logicol_mtx (pd.DataFrame), which is also saved to the out/ directory as a .csv.
    """

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory {root} could not be found.")

    if not os.path.isdir(out):
        os.makedirs(out)

    edf_files = []
    unreadable_files = []

    # collect edf files under root directory
    for path, subdirs, files in os.walk(root):
        for file in files:
            if file.split(".")[-1].lower() in ["edf", "edf+", "bdf", "bdf+"]:
                edf_files.append(os.path.join(path, file))

    # read headers and store in dict of filepath -> header
    edf_headers = {}
    for file in edf_files:
        try:
            edf_headers[file] = pyedflib.highlevel.read_edf_header(file, read_annotations=False)
        except OSError as e:
            print(f"\tCould not read {file} ({e}), so removing from further processing.", enabled=constants.VERBOSE)
            unreadable_files.append(file)

    # remove any files that we couldn't read
    for file in unreadable_files:
        edf_files.remove(file)

    """
    # use headers/signalheaders to pick a sample rate
    edf_sample_rates, sr_cutoff, sr_outlier_files = all_header_indicated_sample_rates(edf_headers)
    edf_sample_rate = edf_sample_rates[0]  # use most frequent as global sample rate
    # TODO resampling - it HAS TO BE DONE BY THIS POINT!

    # remove any files judged as outlier based on sample rate
    for file in sr_outlier_files:
        edf_files.remove(file)
    """

    # sort edf_files by start date
    edf_files = sorted(edf_files, key=lambda x: edf_headers[x]["startdate"])

    # rearrange edf_headers now that we've sorted files
    edf_headers = {file: edf_headers[file] for file in edf_files}

    # using headers, construct matrix of which channels present in which files
    edf_channels_dict = {file: header["channels"] for file, header in edf_headers.items()}
    edf_channels_superset = sorted(set.union(*map(set, edf_channels_dict.values())))
    edf_channels_ndarray = np.zeros(shape=(len(edf_channels_superset), len(edf_files)), dtype=np.int8)
    for i, file in enumerate(edf_files):
        for logicol_channel_intervals, channel_label in enumerate(edf_channels_superset):
            # if the channel is present in this file, set corresponding matrix pos to 1
            if channel_label in edf_headers[file]["channels"]:
                edf_channels_ndarray[logicol_channel_intervals, i] = 1
    edf_channels_mtx = pd.DataFrame(edf_channels_ndarray, index=edf_channels_superset, columns=edf_files, dtype=np.int8)
    edf_channels_mtx.to_csv(os.path.join(out, constants.CHANNEL_PRESENCE_MATRIX_FILENAME), index_label="index")

    # for each channel, get the number of files it appears in
    edf_channels_counts = {channel:np.sum(edf_channels_mtx.loc[channel]) for channel in edf_channels_superset}
    # use this info to determine channels we will later exclude
    min_channel_count = len(edf_files) * 0.75
    excluded_channels = [channel for channel, count in edf_channels_counts.items() if count < min_channel_count]
    if len(excluded_channels) > 0:
        print("Channels {} will be excluded, as they are in less than 75% of {} files (counts {}, 75% thresh: {})".format(
                excluded_channels, len(edf_files),
                [edf_channels_counts[channel] for channel in excluded_channels], min_channel_count),
            enabled=constants.VERBOSE)
    edf_channels_superset = [ch for ch in edf_channels_superset if ch not in excluded_channels]

    # create dict of {file -> {channel -> sample_rate}}
    edf_channel_sample_rates = {}
    for file, header in edf_headers.items():
        edf_channel_sample_rates[file] = {}
        for sig_header in header["SignalHeaders"]:
            edf_channel_sample_rates[file][sig_header["label"]] = sig_header["sample_rate"]

    # using headers, collect start/end times for each file
    edf_intervals = {}
    for file in edf_files:
        start = edf_headers[file]["startdate"]
        end = start + datetime.timedelta(seconds=edf_headers[file]["Duration"])

        edf_intervals[file] = (start, end)


    # produce the start/end times and duration of the logical collation (logicol) of the EDF files
    # edf_compile program will use our logicol data to produce a physical collation (physicol) (i.e, big EDF file)
    logicol_start_dt = min([interval[0] for interval in edf_intervals.values()])
    logicol_end_dt = max([interval[1] for interval in edf_intervals.values()])
    logicol_dur_dt = logicol_end_dt - logicol_start_dt

    """
    logicol_start_samples = 0
    logicol_dur_samples = math.floor(logicol_dur_dt.total_seconds()) * edf_sample_rate
    logicol_end_samples = logicol_dur_samples
    """

    # work out where each file starts/ends (s) in the logical collation
    # i.e, if we were to stitch the EDF files together into one, accounting for gaps, from which & to which second
    #   would each file occupy, not yet assuming overlaps are possible
    logicol_file_intervals = {}
    for file, interval in edf_intervals.items():

        file_start_dt = interval[0]
        file_end_dt = interval[1]
        file_dur_dt = file_end_dt - file_start_dt
        file_dur_s = file_dur_dt.total_seconds()

        logicol_file_start_dt = file_start_dt - logicol_start_dt
        # how far, in s, is file start time from logicol start?
        logicol_file_start_s = logicol_file_start_dt.total_seconds()
        logicol_file_end_s = logicol_file_start_s + file_dur_s

        # convert to int
        logicol_file_start_s = int(np.floor(logicol_file_start_s))
        logicol_file_end_s = int(np.floor(logicol_file_end_s))

        logicol_file_intervals[file] = (logicol_file_start_s, logicol_file_end_s)


    # for each file, logical collation position for each channel (so same as logicol intervals, but greater resolution)
    # these entries will indicate where in the collation of files channels start and stop
    # ideally the same for each channel in each file, but facility provided to adjust on per-channel basis
    #   e.g, if 2 files overlap, but only in the ECG channel etc
    # note we do not yet consider overlaps; intervals in this dict are allowed to overlap.
    #   edf_overlap_resolve will use the data in this dict to find overlaps, and trim channel start/ends accordingly.
    logicol_channel_intervals = {}
    for file, channels in edf_channels_dict.items():
        logicol_channel_intervals[file] = {}

        for channel_label in channels:

            logicol_channel_intervals[file][channel_label] = {

                # initialise start/end per channel in file as the same
                # in an ideal world, this remains so, but we may have to adjust channels within files for overlaps
                "collated_start":   logicol_file_intervals[file][0],
                "collated_end":     logicol_file_intervals[file][1],

                # this remains the same, regardless of whether we adjust channel for overlaps.
                # length of this channel data as-is, if we turned it into an array
                "channel_duration":   logicol_file_intervals[file][1] - logicol_file_intervals[file][0]
            }


    # assign increasing numerical value to each file to indicate its position
    logicol_file_positions = dict(zip(logicol_file_intervals.keys(), range(0, len(logicol_file_intervals))))

    # convert logicol_channel_intervals to a data frame and output as CSV, for use by other programs
    logicol_mtx_entries = []
    logicol_mtx_entry = {
        "file":             None,  # what file is this channel in?
        "file_pos":         None,  # what position is this file in collated set?
        "channel":          None,  # what channel are we looking at?
        "collated_start":   None,  # what index in collated file does this channel start at?
        "collated_end":     None,
        "channel_duration":   None,  # what is the length of this channel, regardless of whether we trim it for overlaps?
        "channel_sample_rate": None,
    }
    for file, channels in logicol_channel_intervals.items():

        file_pos = logicol_file_positions[file]

        for channel, channel_details in channels.items():
            channel_entry = logicol_mtx_entry.copy()

            channel_entry["file"] = file
            channel_entry["file_pos"] = file_pos
            channel_entry["channel"] = channel
            channel_entry["collated_start"] = channel_details["collated_start"]
            channel_entry["collated_end"] = channel_details["collated_end"]
            channel_entry["channel_duration"] = channel_details["channel_duration"]
            channel_entry["channel_sample_rate"] = edf_channel_sample_rates[file][channel]

            logicol_mtx_entries.append(channel_entry)

    # convert to mtx
    logicol_mtx = pd.DataFrame([entry for entry in logicol_mtx_entries])

    # remove excluded channels
    logicol_mtx = logicol_mtx[~logicol_mtx["channel"].isin(excluded_channels)]

    # save to csv
    logicol_mtx.to_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_CHECK_FILENAME), index_label="index")

    # save details like root dir and channels list to a details .json file, for use by other scripts
    with open(os.path.join(out, constants.DETAILS_JSON_FILENAME), "w") as details_file:
        details = {
            "root": root,
            "channels_superset": edf_channels_superset,
        }
        json.dump(details, details_file)


    return logicol_mtx




