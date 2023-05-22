import datetime
import enum
import os
import itertools
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyedflib

from .utils import constants
from .utils.custom_print import print


class OverlapType(enum.Enum):

    NO_OVERLAP = 1,

    # e.g:
    # fa:   fa_start---------------->fa_end
    # fb:              fb_start------------ ...
    PARTIAL_BOTH_ENDOF_A = 2,  # the situation above
    PARTIAL_BOTH_ENDOF_B = 3,  # the situation above but flipped

    # e.g:
    # fa:           fa_start-------->fa_end
    # fb:  fb_start--------------------------->fb_end
    ENTIRETY_FILE_A = 4,  # the situation above
    ENTIRETY_FILE_B = 5,  # the situation above but flipped

    # e.g:
    # fa:  fa_start----------------->fa_end
    # fb:  fb_start----------------->fb_end
    ENTIRETY_BOTH_FILES = 6


def check_time_overlap(t1_start, t1_end, t2_start, t2_end):
    """
    Return Overlap type and overlap start/end time and duration if two recordings overlap in time, otherwise No Overlap and None.
    """

    # thanks to: https://chandoo.org/wp/date-overlap-formulas/
    # this method relies on the ease datetime comparison in Python, e.g datetime(3pm) > datetime(2pm) is TRUE

    # t1_start, t1_end should be the start of the file we're interested in (f1)
    # i.e we want to know how f2 overlaps relative to f1

    # check first if they do overlap
    # in reality, wouldn't make sense to have <= here; would both be <.
    # but, we have determined that duration start is inclusive, end is exclusive.
    # so if a file starts as another one ends; actually, it starts exactly after it ends.
    # TODO; are the 3 lines above correct

    # easy to quickly check that the intervals AREN't overlapping.
    overlap = not ((t1_end <= t2_start) or (t2_end <= t1_start))

    if overlap:  # then determine how they overlap

        type = OverlapType.NO_OVERLAP

        # f1 and f2 line up perfectly (likely a duplicate) i.e
        # f1:  t1_start----------------->t1_end
        # f2:  t2_start----------------->t2_end
        if (t1_start == t2_start) and (t1_end == t2_end):

            type = OverlapType.ENTIRETY_BOTH_FILES
            t3_start = t2_start
            t3_end = t2_end

        # f2 starts (either at same time as f1 or later), but before f1 finishes. e.g:
        # f1:   t1_start---------------->t1_end
        # f2:              t2_start------------ ...
        elif ((t2_start >= t1_start) and (t2_start < t1_end)) or (t2_start == t1_end):

            type = OverlapType.PARTIAL_BOTH_ENDOF_A
            t3_start = t2_start

            # does f2 start and end during f1? e.g
            # f1:  t1_start--------------------------->t1_end
            # f2:           t2_start-------->t2_end
            if ((t2_end > t1_start) and (t2_end < t1_end)):
                type = OverlapType.ENTIRETY_FILE_B
                t3_end = t2_end
            else:
                t3_end = t1_end

        # f1 starts (either at the same time as f2 or later), but before f2 finishes, e.g
        # f1:           t1_start--------------- ...
        # f2:  t2_start--------------->t2_end
        elif ((t1_start >= t2_start) and (t1_start < t2_end)) or (t1_start == t2_end):

            type = OverlapType.PARTIAL_BOTH_ENDOF_B
            t3_start = t1_start

            # does f1 start and end during f2? e.g
            # f1:           t1_start-------->t1_end
            # f2:  t2_start--------------------------->t2_end
            if ((t1_end > t2_start) and (t1_end < t2_end)):
                type = OverlapType.ENTIRETY_FILE_A
                t3_end = t1_end
            else:
                t3_end = t2_end


        else:

            # This shouldn't occur, but left in just in-case.
            raise Exception("""ERROR: overlap type not dealt with
                                            \nt1 = {} -> {}, \nt2 = {} -> {}"""
                            .format(t1_start, t1_end, t2_start, t2_end))

        return type, (t3_start, t3_end, (t3_end - t3_start))# + datetime.timedelta(seconds=1))

    else:
        return OverlapType.NO_OVERLAP, None

def check_channel_overlap(file_a_header, file_b_header):
    """Return lists of common and unique channels between two EDF files, or an empty list if there are none in common. """

    file_a_channels = file_a_header["channels"]
    file_b_channels = file_b_header["channels"]

    # if no channels in common, UNLIKELY there is any overlap
    #   - in theory, channels with same data could have been renamed somehow.
    #       - BUT so unlikely, and would require checking all the data of potentially many files, which will be SLOW
    common_channels = [channel for channel in file_a_channels if channel in file_b_channels]
    if len(common_channels) == 0:
        return []

    file_a_unique_channels = [channel for channel in file_a_channels if channel not in file_b_channels]
    file_b_unique_channels = [channel for channel in file_b_channels if channel not in file_a_channels]

    return [common_channels, file_a_unique_channels, file_b_unique_channels]


def interval_plot(interval_list):

    fig, ax = plt.subplots()

    for i in range(0, len(interval_list)):
        start = interval_list[i][0]
        end = interval_list[i][1]
        ax.plot([start, end], [-i, -i])

    return fig, ax



def check(root, out, mtx=None, verbose=constants.VERBOSE):

    print("edf_overlaps.check: Checking for overlaps (this may take a while on larger datasets) ... ", enabled=verbose)

    # create a data structure to hold information about an overlap
    overlap_mtx_entries = []
    overlap_mtx_entry = {
        "channel":              None,
        "file_A":               None,
        "file_B":               None,
        "overlap_start":        None,
        "overlap_end":          None,
        "overlap_duration":     None,
        "overlap_type":         None,
        #"data_identical":       None, # bool
        #"action_taken":         None,
    }


    # edf_check_overlap will provide an overlap-adjusted mtx we should use instead
    if isinstance(mtx, pd.DataFrame):

        # try to ensure provided mtx is of correct format
        logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_RESOLVE_FILENAME),
                                  index_col="index")
        if list(mtx.columns) == list(logicol_mtx.columns):
            logicol_mtx = mtx
        else:
            raise TypeError("edf_overlaps.check: Matrix provided does not appear to be similar to logicol_mtx")
    else:

        try:
            logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_POST_OVERLAP_RESOLVE_FILENAME),
                                      index_col="index")
            print("edf_overlaps.check: Overlap-trimmed Logicol Matrix found and successfully loaded", enabled=verbose)

        except FileNotFoundError:

            try:
                logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_RESOLVE_FILENAME),
                                          index_col="index")

            except FileNotFoundError:
                print(
                    f"edf_overlaps.check: Warning: Logicol Matrix could not be found in {out} - have you run edf_collate?",
                    enabled=True)
                sys.exit(os.EX_NOINPUT)


    all_channels = pd.unique(logicol_mtx["channel"])

    for channel in all_channels:

        # get rows representing this channel across files
        this_channel = logicol_mtx[logicol_mtx["channel"] == channel]

        # produce every unique pair of these rows (using their index)
        row_combinations = list(itertools.combinations(this_channel.index, 2))

        for pair in row_combinations:

            row_A = this_channel.loc[pair[0]]
            row_B = this_channel.loc[pair[1]]

            overlap_type, overlap_period = check_time_overlap(row_A["collated_start"], row_A["collated_end"],
                                                              row_B["collated_start"], row_B["collated_end"],)

            if overlap_type != OverlapType.NO_OVERLAP:
                #print(f"{str(overlap_type), str(overlap_period)}", enabled=constants.VERBOSE)

                overlap_entry = overlap_mtx_entry.copy()

                overlap_entry["channel"] = channel
                overlap_entry["file_A"] = row_A["file"]
                overlap_entry["file_B"] = row_B["file"]
                overlap_entry["overlap_start"] = overlap_period[0]
                overlap_entry["overlap_end"] = overlap_period[1]
                overlap_entry["overlap_duration"] = overlap_period[2]
                overlap_entry["overlap_type"] = overlap_type

                overlap_mtx_entries.append(overlap_entry)

    # TODO do we want to save overlap_mtx? current only used for verbose
    overlap_mtx = pd.DataFrame([entry for entry in overlap_mtx_entries])

    if verbose:
        n_overlaps = overlap_mtx.shape[0]
        if n_overlaps == 0:
            print("edf_overlaps.check: No overlaps detected.", enabled=verbose)
        else:
            n_overlapping_channels = len(pd.unique(overlap_mtx["channel"]))
            n_channels_total = len(pd.unique(logicol_mtx["channel"]))

            n_overlapping_files = overlap_mtx.groupby(["file_A", "file_B"]).ngroups
            n_files_total = len(pd.unique(logicol_mtx["file"]))
            print(f"edf_overlaps.check: {n_overlaps} overlapping channels ({n_overlapping_channels}/{n_channels_total} unique channels total) across {n_overlapping_files}/{n_files_total} pairs of files!", enabled=verbose)


            types = np.array([overlap["overlap_type"] for overlap in overlap_mtx_entries])
            unique_types = []
            for type in types:
                if type not in unique_types:
                    unique_types.append(type)
            types_counts = dict(zip(unique_types, [np.sum(types == type) for type in unique_types]))
            print(f"Unique overlap types: {[str(type) for type in unique_types]}", enabled=verbose)
            for type, count in types_counts.items():
                print(f"{str(type)}: {count} occurrences.", enabled=verbose)
            print("\n", enabled=verbose)
            #print(pd.DataFrame([overlap["overlap_type"] for overlap in overlap_mtx_entries]).value_counts(), enabled=True)

    return overlap_mtx_entries

def single_excluded_channel(out, excluded_channel, reason):
    excluded_channels_filename = os.path.join(out, constants.EXCLUDED_CHANNELS_LIST_FILENAME)

    excluded_channel_copy = excluded_channel.copy()
    excluded_channel_copy["reason"] = reason

    try:
        excluded_channels = pd.read_csv(excluded_channels_filename, index_col="index")
        excluded_channels = pd.concat([excluded_channels, excluded_channel_copy])
    except FileNotFoundError:
        excluded_channels = excluded_channel_copy

    excluded_channels.to_csv(excluded_channels_filename, index_label="index")

def resolve(root, out):

    print("edf_overlaps.resolve: Resolving overlaps (This *will* take some time - but only needs to be done once.) ... ", enabled=constants.VERBOSE)

    # logicol_mtx holds information on where channels start/end in logical collation, though overlaps may be present
    try:
        logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_POST_OVERLAP_RESOLVE_FILENAME),
                                       index_col="index")
        print(
            f"edf_overlaps.resolve: Warning: Overlap-trimmed Logicol Matrix already found in {out} - it is unlikely you need to re-run this program again.\n"
            f"edf_overlaps.resolve: Continue anyway (re-generate trimmed Logicol Matrix)? (y/n)", enabled=True)
        if str(input("> ")).lower() == "y":
            logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_RESOLVE_FILENAME),
                                      index_col="index")
        else:
            return

    except FileNotFoundError:

        try:
            logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_RESOLVE_FILENAME), index_col="index")

        except FileNotFoundError:
            print(f"edf_overlaps.resolve: Warning: Logicol Matrix could not be found in {out} - have you run edf_collate?", enabled=True)
            sys.exit(os.EX_NOINPUT)



    # first, check if we actually have any overlaps to resolve
    overlaps = check(root, out, mtx=logicol_mtx, verbose=False)

    if len(overlaps) == 0:
        print("edf_overlaps.resolve: No overlaps found to be present, no further action will be taken.", enabled=constants.VERBOSE)
        return

    # we will trim overlaps from this mtx in this method, if there are any
    logicol_mtx_trimmed = logicol_mtx.copy()


    # create a data structure to hold information about an overlap
    resolution_mtx_entries = []
    resolution_mtx_entry = {
        "channel":              None,
        "file_A":               None,
        "file_B":               None,
        "overlap_start":        None,
        "overlap_end":          None,
        "overlap_duration":     None,
        "overlap_type":         None,

        "data_identical":       None,
        "action_taken":         None,
    }

    # loop terminates when all overlaps resolved
    while True:

        if len(overlaps) == 0:
            print(f"edf_overlaps.resolve: All overlaps resolved! See {os.path.join(out, constants.EXCLUDED_CHANNELS_LIST_FILENAME)} for info on channels omitted/trimmed. You won't need to run this again unless data in root dir changes.", enabled=constants.VERBOSE)
            break

        # print progress
        print(f"\redf_overlaps.resolve: {len(overlaps)} overlaps left to resolve.", end="", enabled=constants.VERBOSE)

        # resolve first overlap, then re-call check_overlaps; if resolved, it will not be present, and we move on to next
        overlap = overlaps[0]

        # get the logicol info for these channels
        file_a_logicol = logicol_mtx_trimmed.loc[
            (logicol_mtx_trimmed["file"] == overlap["file_A"]) & (logicol_mtx_trimmed["channel"] == overlap["channel"])]
        file_b_logicol = logicol_mtx_trimmed.loc[
            (logicol_mtx_trimmed["file"] == overlap["file_B"]) & (logicol_mtx_trimmed["channel"] == overlap["channel"])]


        file_a_collated_start = file_a_logicol["collated_start"].item()
        file_a_collated_end = file_a_logicol["collated_end"].item()

        file_b_collated_start = file_b_logicol["collated_start"].item()
        file_b_collated_end = file_b_logicol["collated_end"].item()

        overlap_duration = overlap["overlap_duration"]

        # open handles to access file data
        file_a = pyedflib.EdfReader(overlap["file_A"], 0, 1)
        file_b = pyedflib.EdfReader(overlap["file_B"], 0, 1)

        # read the overlapping data from both files
        # i.e, where does the overlap start/end within each file (in seconds, w.r.t start of each file (0))

        # e.g:
        # fa:   fa_start---------------->fa_end
        # fb:              fb_start------------ ...
        if overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_A:

            file_a_overlap_start = file_b_collated_start - file_a_collated_start
            file_b_overlap_start = 0

        elif overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_B: # maybe impossible?

            raise NotImplementedError(f"Reading overlapping data, type is {OverlapType.PARTIAL_BOTH_ENDOF_B}")

        elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_A: # maybe impossible?

            raise NotImplementedError(f"Reading overlapping data, type is {OverlapType.ENTIRETY_FILE_A}")

        elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_B:

            raise NotImplementedError(f"Reading overlapping data, type is {OverlapType.ENTIRETY_FILE_B}")

        elif overlap["overlap_type"] is OverlapType.ENTIRETY_BOTH_FILES:

            file_a_overlap_start = 0
            file_b_overlap_start = 0      

        file_a_overlap_end = file_a_overlap_start + overlap_duration
        file_b_overlap_end = file_b_overlap_start + overlap_duration

        file_a_sample_rate = file_a_logicol["channel_sample_rate"].item()
        file_a_channel_idx = file_a.getSignalLabels().index(overlap["channel"])
        file_a_overlap_data = file_a.readSignal(file_a_channel_idx,
                                                start=int(np.floor((file_a_overlap_start * file_a_sample_rate))),
                                                n = int(np.floor((overlap_duration * file_a_sample_rate))))


        file_b_sample_rate = file_b_logicol["channel_sample_rate"].item()
        file_b_channel_idx = file_b.getSignalLabels().index(overlap["channel"])
        file_b_overlap_data = file_b.readSignal(file_b_channel_idx,
                                                start=int(np.floor((file_b_overlap_start * file_b_sample_rate))),
                                                n=int(np.floor((overlap_duration * file_b_sample_rate))))

        file_a.close()
        file_b.close()

        if file_a_sample_rate != file_b_sample_rate:
            raise NotImplementedError(f"Data has differing sample rates: A={file_a_sample_rate}Hz, B={file_b_sample_rate}Hz")

        # is overlapping data the same?
        # TODO what about sample rates?
            # surely if differing sample rate, data wont be the same.
            # keep data of higher sample rate?
        # TODO move open excluded_channels csv functionality into function, add excluded channels when data is identical
        if all(file_a_overlap_data == file_b_overlap_data):

            if overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_A:
                
                # trim the overlapping data from the end of the channel in file A
                logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] = file_b_logicol["collated_start"].item()
                single_excluded_channel(out, file_a_logicol, f"Overlaps partially with {file_b_logicol.index.item()}, but data is the same; identical section trimmed from end of this channel.")

            elif overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_B:

                raise NotImplementedError(f"Data is the same, type is {OverlapType.PARTIAL_BOTH_ENDOF_B}")

            elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_A:

                raise NotImplementedError(f"Data is the same, type is {OverlapType.ENTIRETY_FILE_A}")

            elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_B:

                raise NotImplementedError(f"Data is the same, type is {OverlapType.ENTIRETY_FILE_B}")

            elif overlap["overlap_type"] is OverlapType.ENTIRETY_BOTH_FILES:

                # same data occurs at same time in 2 different files, for entirety of each file, so we can arbitrarily remove one 
                logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] = file_a_logicol["collated_start"].item()
                single_excluded_channel(out, file_a_logicol, f"Overlaps entirely with {file_b_logicol.index.item()}, but data is the same, so this channel was removed.")


        else:  # data isn't the same; more problematic. TODO keep longest file (makes sense if think about it)? even if only temp solution

            if overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_A:

                raise NotImplementedError(f"Data is NOT the same, type is {OverlapType.PARTIAL_BOTH_ENDOF_A}")

            elif overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_B:

                raise NotImplementedError(f"Data is NOT the same, type is {OverlapType.PARTIAL_BOTH_ENDOF_B}")

            elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_A:

                raise NotImplementedError(f"Data is NOT the same, type is {OverlapType.ENTIRETY_FILE_A}")

            elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_B:

                raise NotImplementedError(f"Data is NOT the same, type is {OverlapType.ENTIRETY_FILE_B}")

            elif overlap["overlap_type"] is OverlapType.ENTIRETY_BOTH_FILES:
                # remove both channels 
                logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] = file_a_logicol["collated_start"].item()
                logicol_mtx_trimmed.at[file_b_logicol.index.item(), "collated_end"] = file_b_logicol["collated_start"].item()

                file_a_logicol_copy = file_a_logicol.copy()
                file_a_logicol_copy["reason"] = f"Overlaps entirely with {file_b_logicol.index.item()}, but data is not the same."
                file_b_logicol_copy = file_b_logicol.copy()
                file_b_logicol_copy["reason"] = f"Overlaps entirely with {file_a_logicol.index.item()}, but data is not the same."

                excluded_channels_filename = os.path.join(out, constants.EXCLUDED_CHANNELS_LIST_FILENAME)
                try:
                    excluded_channels = pd.read_csv(excluded_channels_filename, index_col="index")
                    excluded_channels = pd.concat([excluded_channels, file_a_logicol_copy, file_b_logicol_copy])
                except FileNotFoundError:
                    excluded_channels = pd.concat([file_a_logicol_copy, file_b_logicol_copy])

                excluded_channels.to_csv(excluded_channels_filename, index_label="index")
                print(f"edf_overlaps.resolve: Encountered overlap consisting entirely of both channels involved, but data not the same. Both channels omitted. See {excluded_channels_filename}. Manual correction recommended.", enabled=constants.VERBOSE)


        # we've fixed an overlap, so re-generate
        overlaps = check(root, out, mtx=logicol_mtx_trimmed, verbose=False)

    # save trimmed logicol_mtx to csv
    logicol_mtx_trimmed.to_csv(os.path.join(out, constants.LOGICOL_POST_OVERLAP_RESOLVE_FILENAME), index_label="index")



