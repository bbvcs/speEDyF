import os
import datetime
import itertools
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pyedflib

from utils.custom_print import print
from utils.overlap_resolution import OverlapType, check_time_overlap, check_channel_overlap, resolve_overlap, resolve_overlap2, interval_plot
from utils.resampling import all_header_indicated_sample_rates



def edf_collate_OLD(root, out):

    VERBOSE = True  # TODO make param

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
            print(f"\tCould not read {file} ({e})", enabled=VERBOSE)
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
            enabled=VERBOSE)
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
                            print(str(overlap_type), enabled=VERBOSE)

                            resolve_overlap2(overlapping_pair_count, resolution_mtx_entries, edf_channel_intervals, edf_sample_rate,
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
    #         print(str(overlap_type), enabled=VERBOSE)
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


def edf_collate(root, out):

    VERBOSE = True  # TODO make param

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory {root} could not be found.")

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
            print(f"\tCould not read {file} ({e}), so removing from further processing.", enabled=VERBOSE)
            unreadable_files.append(file)

    # remove any files that we couldn't read
    for file in unreadable_files:
        edf_files.remove(file)

    # use headers/signalheaders to pick a sample rate
    edf_sample_rates, sr_cutoff, sr_outlier_files = all_header_indicated_sample_rates(edf_headers)
    edf_sample_rate = edf_sample_rates[0]  # use most frequent as global sample rate
    # TODO resampling - it HAS TO BE DONE BY THIS POINT!

    # remove any files judged as outlier based on sample rate
    for file in sr_outlier_files:
        edf_files.remove(file)

    # sort edf_files by start date
    edf_files = sorted(edf_files, key=lambda x: edf_headers[x]["startdate"])

    # rearrange edf_headers now that we've sorted files
    edf_headers = {file: edf_headers[file] for file in edf_files}

    # using headers, construct matrix of which channels present in which files
    edf_channels_dict = {file: header["channels"] for file, header in edf_headers.items()}
    edf_channels_superset = sorted(set.union(*map(set, edf_channels_dict.values())))
    edf_channels_ndarray = np.zeros(shape=(len(edf_channels_superset), len(edf_files)), dtype=np.int8)
    for i, file in enumerate(edf_files):
        for collation_dict, channel_label in enumerate(edf_channels_superset):
            # if the channel is present in this file, set corresponding matrix pos to 1
            if channel_label in edf_headers[file]["channels"]:
                edf_channels_ndarray[collation_dict, i] = 1
    edf_channels_mtx = pd.DataFrame(edf_channels_ndarray, index=edf_channels_superset, columns=edf_files, dtype=np.int8)

    # for each channel, get the number of files it appears in
    edf_channels_counts = {channel:np.sum(edf_channels_mtx.loc[channel]) for channel in edf_channels_superset}
    min_channel_count = len(edf_files) * 0.75
    excluded_channels = [channel for channel, count in edf_channels_counts.items() if count < min_channel_count]
    if len(excluded_channels) > 0:
        print("Channels {} will be excluded, as they are in less than 75% of {} files (counts {}, 75% thresh: {})".format(
                excluded_channels, len(edf_files),
                [edf_channels_counts[channel] for channel in excluded_channels], min_channel_count),
            enabled=VERBOSE)
    # TODO exclude (set to NaN in collation matrix)

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

    logicol_start_samples = 0
    logicol_dur_samples = math.floor(logicol_dur_dt.total_seconds()) * edf_sample_rate
    logicol_end_samples = logicol_dur_samples


    # work out where each file starts/ends (sample-wise) in the logical collation
    # i.e, if we were to stitch the EDF files together into one, accounting for gaps, from which & to which samples
    #   would each file occupy, not yet assuming overlaps are possible
    logicol_intervals = {}
    for file, interval in edf_intervals.items():

        file_start_dt = interval[0]
        file_end_dt = interval[1]
        file_dur_dt = file_end_dt - file_start_dt
        file_dur_samples = math.floor(file_dur_dt.total_seconds()) * edf_sample_rate

        logicol_file_start_dt = file_start_dt - logicol_start_dt
        # how far, in s, is file start time from logicol start?
        logicol_file_start_samples = math.floor(logicol_file_start_dt.total_seconds()) * edf_sample_rate

        # logicol_file_end_s = logicol_end_dt - file_end_dt
        # logicol_file_end_samples = logicol_file_end_s * edf_sample_rate
        logicol_file_end_samples = logicol_file_start_samples + file_dur_samples

        logicol_intervals[file] = (logicol_file_start_samples, logicol_file_end_samples)


    # initialise collation dict
    # for each file, set up entry for each channel
    # these entries will indicate where in the "theoretical collated" file channels start, stop
    # we don't know where they will start/stop yet; we do this later
    # we do know the raw data length of each channel; intially, this is the same for every channel in a file
    # in an ideal world, this remains so; however, due to overlaps, we might need to cut bits off of the end of only
    #  certain channels in a file.
    # edf_overlap_resolve will use the data in this dict to find overlaps, and trim channel start/ends accordingly.
    # \/\/\/\/\/\/\\/\/\/ remove me
    # collation_dict = {}
    # for file, channels in edf_channels_dict.items():
    #     collation_dict[file] = {}
    #
    #     file_interval = edf_intervals[file]
    #     file_dur_samples = file_interval[1] - file_interval[0]
    #
    #     for channel_label in channels:
    #
    #         collation_dict[file][channel_label] = {
    #             "collated_start" : None,
    #             "collated_end"   : None,
    #             "channel_length" : file_dur_samples
    #         }




    return


if __name__ == "__main__":
    # TODO: add argparse library, pass in root/out
    root = "/home/bcsm/University/stage-3/BSc_Project/program/code/FILES/INPUT_DATA/909"
    #root = "/home/bcsm/University/stage-3/BSc_Project/program/code/FILES/INPUT_DATA/909/test_overlap_3"
    #root = "/home/bcsm/Desktop/895" # partial, non-identical (real)
    # root = "/home/bcsm/University/stage-3/BSc_Project/program/code/FILES/INPUT_DATA/909/test_overlap_partial_nonidentical" # simulated
    #root = "/home/bcsm/Desktop/865" # many partial identical
    #root = "/home/bcsm/University/stage-3/BSc_Project/program/code/FILES/INPUT_DATA/put_my_contents_back_in_909/test_3_shifted_overlap"

    # dir to save results for this run i.e matrix, overlap resolutions
    out = "out/testing"

    edf_collate(root, out)