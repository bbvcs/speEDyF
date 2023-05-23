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

def edf_collate(root, out, minimum_edf_channel_sample_rate_hz=32, forced=False):
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
    :param minimum_edf_channel_sample_rate_hz: channels within files below this sample rate will be excluded.
    :param forced: If false (recommended), check whether we have already collated this subject. Set to true if automating.
    :return: logicol_mtx (pd.DataFrame), which is also saved to the out/ directory as a .csv.
    """

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory {root} could not be found.")

    if not os.path.isdir(out):
        os.makedirs(out)

    if not forced:
        try:
            pd.read_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_RESOLVE_FILENAME), index_col="index")
            """
            print(
                f"edf_collate: Warning: Logicol Matrix already found in {out}. If data in {root} hasn't changed, you are unlikely to need to run this program again.\n"
                f"edf_collate: Re-generate Logicol Matrix? (y/n)", enabled=True)
            if str(input("> ")).lower() != "y":
                return
            """

            print(
                f"edf_collate: Warning: Logicol Matrix already found in {out}. If data in {root} hasn't changed, you are unlikely to need to run this program again.\n"
                f"edf_collate: Parameter forced=False, so the Logicol Matrix will not be re-generated. If you need to re-generate the Logicol Matrix, re-run edf_collate with forced=True.", enabled=True)
            return

        except FileNotFoundError:
            pass


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

    """
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
    """

    # create dict of {file -> {channel -> sample_rate}}
    edf_channel_sample_rates = {}
    for file, header in edf_headers.items():
        edf_channel_sample_rates[file] = {}
        for sig_header in header["SignalHeaders"]:
            channel = sig_header["label"]
            sample_rate = sig_header["sample_rate"]

            edf_channel_sample_rates[file][channel] = sample_rate



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


    excluded_channels = logicol_mtx[logicol_mtx["channel_sample_rate"] < minimum_edf_channel_sample_rate_hz]
    excluded_channels_filename = os.path.join(out, constants.EXCLUDED_CHANNELS_LIST_FILENAME)
    if len(excluded_channels) > 0:
        reasons = pd.Series(np.repeat(f"sample rate below specified minimum {minimum_edf_channel_sample_rate_hz} Hz", excluded_channels.shape[0]))
        excluded_channels = excluded_channels.assign(reason=reasons.values)
        excluded_channels.to_csv(excluded_channels_filename, index_label="index")
        print(f"edf_collate: {len(excluded_channels)} channels (names: {pd.unique(excluded_channels['channel'])}) have "
              f"been excluded, as their sample rates were below specified minimum {minimum_edf_channel_sample_rate_hz} Hz. "
              f"Full details can be found in {excluded_channels_filename}.", enabled=constants.VERBOSE)

    # remove channels below user-defined minimum sample rate
    logicol_mtx = logicol_mtx[logicol_mtx["channel_sample_rate"] >= minimum_edf_channel_sample_rate_hz]
    edf_channels_superset = list(pd.unique(logicol_mtx["channel"])) # update channels superset accordingly

    # save to csv
    logicol_mtx.to_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_RESOLVE_FILENAME), index_label="index")

    # save details like root dir and channels list to a details .json file, for use by other scripts
    with open(os.path.join(out, constants.DETAILS_JSON_FILENAME), "w") as details_file:
        details = {
            "root": root,
            "channels_superset": edf_channels_superset,
        }
        json.dump(details, details_file)


    return logicol_mtx




