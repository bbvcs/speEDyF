import datetime
import enum
import os
import itertools
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

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
    """Add details for an excluded/trimmed channel to the excluded_channels.csv"""
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
            print(f"\nedf_overlaps.resolve: All overlaps resolved! See {os.path.join(out, constants.EXCLUDED_CHANNELS_LIST_FILENAME)} for info on channels omitted/trimmed. You won't need to run this again unless data in root dir changes.", enabled=constants.VERBOSE)

            # has the start/end date changed after trimming?
            with open(os.path.join(out, constants.DETAILS_JSON_FILENAME), "r") as details_file:
                details = json.load(details_file)

            start_header = pyedflib.highlevel.read_edf_header(logicol_mtx_trimmed.iloc[0]["file"], read_annotations=False)
            end_header = pyedflib.highlevel.read_edf_header(logicol_mtx_trimmed.iloc[-1]["file"], read_annotations=False)

            startdate = str(start_header["startdate"])
            enddate = str(end_header["startdate"] + datetime.timedelta(seconds=end_header["Duration"]))

            changed = False

            if startdate != details["startdate"]:
                details["startdate"] = startdate
                changed = True
            if enddate != details["enddate"]:
                details["enddate"] = enddate
                changed = True

            if changed:
                # update details
                with open(os.path.join(out, constants.DETAILS_JSON_FILENAME), "w") as details_file:
                    json.dump(details, details_file)

            break

        # print progress
        print(f"\redf_overlaps.resolve: {len(overlaps)} overlaps left to resolve.", end="", enabled=constants.VERBOSE)

        # resolve first overlap, then re-call check_overlaps; if resolved, it will not be present, and we move on to next
        overlap = overlaps[0]

        # get the logicol info for these channels TODO these should be called channel_a_logicol etc not file_a_logicol
        # if the file is the same & channel is the same & they overlap in time with the overlap object (see def check_time_overlap()). This last part may seem redundant, but resolving ENTIRETY FILE B can split a channel in 2, so without last clause would get 2 hits 
        file_a_logicol = logicol_mtx_trimmed.loc[(logicol_mtx_trimmed["file"] == overlap["file_A"]) & (logicol_mtx_trimmed["channel"] == overlap["channel"]) & (~((logicol_mtx_trimmed["collated_end"] <= overlap["overlap_start"]) | (overlap["overlap_end"]<= logicol_mtx_trimmed["collated_start"])))]
        file_b_logicol = logicol_mtx_trimmed.loc[(logicol_mtx_trimmed["file"] == overlap["file_B"]) & (logicol_mtx_trimmed["channel"] == overlap["channel"]) & (~((logicol_mtx_trimmed["collated_end"] <= overlap["overlap_start"]) | (overlap["overlap_end"]<= logicol_mtx_trimmed["collated_start"])))]


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

            raise NotImplementedError(f"Reading overlapping data, type is {OverlapType.PARTIAL_BOTH_ENDOF_B}, files: {overlap['file_A']}, {overlap['file_B']}")

        elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_A:
            # empirically, only possible if start of file A has been trimmed as it overlapped with the end of another
            # file, such that it now starts AFTER file B ?

            file_a_overlap_start = 0
            file_b_overlap_start = file_a_collated_start - file_b_collated_start

        elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_B:

            file_a_overlap_start = file_b_collated_start - file_a_collated_start
            file_b_overlap_start = 0

        elif overlap["overlap_type"] is OverlapType.ENTIRETY_BOTH_FILES:

            file_a_overlap_start = 0
            file_b_overlap_start = 0      

        file_a_overlap_end = file_a_overlap_start + overlap_duration
        file_b_overlap_end = file_b_overlap_start + overlap_duration

        file_a_sample_rate = file_a_logicol["channel_sample_rate"].item()
        file_a_signal_labels = [label.upper().strip() for label in file_a.getSignalLabels()]
        file_a_channel_idx = file_a_signal_labels.index(overlap["channel"])
        file_a_overlap_data = file_a.readSignal(file_a_channel_idx,
                                                start=int(np.floor((file_a_overlap_start * file_a_sample_rate))),
                                                n = int(np.floor((overlap_duration * file_a_sample_rate))))


        file_b_sample_rate = file_b_logicol["channel_sample_rate"].item()
        file_b_signal_labels = [label.upper().strip() for label in file_b.getSignalLabels()]
        file_b_channel_idx = file_b_signal_labels.index(overlap["channel"])
        file_b_overlap_data = file_b.readSignal(file_b_channel_idx,
                                                start=int(np.floor((file_b_overlap_start * file_b_sample_rate))),
                                                n=int(np.floor((overlap_duration * file_b_sample_rate))))

        file_a.close()
        file_b.close()

        if file_a_sample_rate != file_b_sample_rate:
            # see edf_segment for how to fix this. Leaving it for now so we get some test data.
            raise NotImplementedError(f"Data has differing sample rates: A={file_a_sample_rate}Hz, B={file_b_sample_rate}Hz")

        # is overlapping data the same?
        # TODO what about sample rates?
            # surely if differing sample rate, data wont be the same.
            # keep data of higher sample rate?

        if all(file_a_overlap_data == file_b_overlap_data):

            if overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_A:
                
                # trim the overlapping data from the end of the channel in file A
                logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] = file_b_logicol["collated_start"].item()
                single_excluded_channel(out, file_a_logicol, f"TRIMMED: Overlaps partially with {file_b_logicol.index.item()}, but data is the same; identical section trimmed from end of this channel.")

            elif overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_B:

                raise NotImplementedError(f"Data is the same, type is {OverlapType.PARTIAL_BOTH_ENDOF_B}, files: {overlap['file_A']}, {overlap['file_B']}")

            elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_A:

                # same data occurs at same time in a and b, but there is more data in b besides the overlapping section, so remove a.
                logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=file_a_logicol.index.item())
                single_excluded_channel(out, file_a_logicol,
                                        f"REMOVED: Overlaps entirely (and data is the same) with {file_b_logicol.index.item()}, but there is data in the other channel besides the overlapping section, so this channel was removed.")


            elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_B:

                # same data occurs at same time in a and b, but there is more data in a besides the overlapping section, so remove b.
                logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=file_b_logicol.index.item())
                single_excluded_channel(out, file_b_logicol,
                                        f"REMOVED: Overlaps entirely (and data is the same) with {file_a_logicol.index.item()}, but there is data in the other channel besides the overlapping section, so this channel was removed.")

            elif overlap["overlap_type"] is OverlapType.ENTIRETY_BOTH_FILES:

                # same data occurs at same time in 2 different files, for entirety of each file, so we can arbitrarily remove one 
                #logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] = file_a_logicol["collated_start"].item()
                logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=file_a_logicol.index.item())
                single_excluded_channel(out, file_a_logicol, f"REMOVED: Overlaps entirely with {file_b_logicol.index.item()}, but data is the same, so this channel was removed.")


        else:  # data isn't the same; more problematic.

            # first, check if either signal appears to be electrical noise
            fa, pxxa = signal.welch(file_a_overlap_data - np.mean(file_a_overlap_data),
                                    file_a_sample_rate, nperseg=file_a_sample_rate * 30)

            # powerline interference is at 50hz/60hz depending on region
            fa_50hz_loc = np.where(fa >= 50)[0][0]
            fa_60hz_loc = np.where(fa >= 60)[0][0]

            # what is the power of the (incl. surrounding freqs) dominant powerline frequency?
            pxxa_50hz_sum = np.sum(pxxa[fa_50hz_loc - 5:fa_50hz_loc + 5])
            pxxa_60hz_sum = np.sum(pxxa[fa_60hz_loc - 5:fa_60hz_loc + 5])

            # what proportion of overall signal power does the dominant powerline frequency occupy?
            file_a_50hz_proportion = pxxa_50hz_sum / np.sum(pxxa)
            file_a_60hz_proportion = pxxa_60hz_sum / np.sum(pxxa)

            fb, pxxb = signal.welch(file_b_overlap_data - np.mean(file_b_overlap_data),
                                    file_b_sample_rate, nperseg=file_b_sample_rate * 30)

            fb_50hz_loc = np.where(fb >= 50)[0][0]
            fb_60hz_loc = np.where(fb >= 60)[0][0]

            pxxb_50hz_sum = np.sum(pxxb[fb_50hz_loc - 5:fb_50hz_loc + 5])
            pxxb_60hz_sum = np.sum(pxxb[fb_60hz_loc - 5:fb_60hz_loc + 5])

            file_b_50hz_proportion = pxxb_50hz_sum / np.sum(pxxb)
            file_b_60hz_proportion = pxxb_60hz_sum / np.sum(pxxb)

            powerline_proportion_threshold = 0.90 # what percentage of signal has to be powerline for us to discard it?

            # what we do with this information depends on type of overlap ...

            if overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_A:

                if (file_a_50hz_proportion > powerline_proportion_threshold) \
                        or (file_a_60hz_proportion > powerline_proportion_threshold):

                    # overlapping section of channel in file A looks to be mostly powerline noise - trim it
                    logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] \
                        = file_b_logicol["collated_start"].item()
                    single_excluded_channel(out, file_a_logicol, f"TRIMMED: Overlaps partially with {file_b_logicol.index.item()}, and data is not the same; this appears to be only electrical powerline noise, so overlapping section trimmed from this channel.")

                elif (file_b_50hz_proportion > powerline_proportion_threshold) \
                        or (file_b_60hz_proportion > powerline_proportion_threshold):

                    # overlapping section of channel in file B looks to be mostly powerline noise - trim it
                    logicol_mtx_trimmed.at[file_b_logicol.index.item(), "collated_start"] \
                        = file_a_logicol["collated_end"].item()
                    single_excluded_channel(out, file_a_logicol, f"TRIMMED: Overlaps partially with {file_a_logicol.index.item()}, and data is not the same; this appears to be only electrical powerline noise, so overlapping section trimmed from this channel.")


                else:

                    # keep the longest channel. TODO Not sure this is best solution but will do for now.
                    # known subjects where this occurs: 1006 (defo not best solution here)
                    # need to check excluded channels mtx to determine other channels where this happens

                    file_a_duration = file_a_logicol["channel_duration"].item()
                    file_b_duration = file_b_logicol["channel_duration"].item()

                    if file_a_duration > file_b_duration:
                        # trim the overlapping data from the start of the channel in file B
                        logicol_mtx_trimmed.at[file_b_logicol.index.item(), "collated_start"] = file_a_logicol["collated_end"].item()
                        single_excluded_channel(out, file_b_logicol,
                                                f"TRIMMED: Overlaps partially with {file_a_logicol.index.item()}. "
                                                f"Data is not the same. Other channel is longer, so overlapping section from end of this channel trimmed.")

                    else:
                        # trim the overlapping data from the end of the channel in file A
                        logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] = file_b_logicol["collated_start"].item()
                        single_excluded_channel(out, file_a_logicol,
                                                f"TRIMMED: Overlaps partially with {file_b_logicol.index.item()}. "
                                                f"Data is not the same. Other channel is longer, so overlapping section from end of this channel trimmed.")



            elif overlap["overlap_type"] is OverlapType.PARTIAL_BOTH_ENDOF_B:

                raise NotImplementedError(f"Data is NOT the same, type is {OverlapType.PARTIAL_BOTH_ENDOF_B}, files: {overlap['file_A']}, {overlap['file_B']}")

            elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_A:

                if (file_b_50hz_proportion > powerline_proportion_threshold) \
                        or (file_b_60hz_proportion > powerline_proportion_threshold):
                    # channel in file B looks to be mostly powerline noise.
                    # file A starts/ends within file B, so trim out the overlapping section from file B, leaving bits on
                    # both sides.

                    j = file_b_logicol.copy()

                    # trim current entry up to the point that the overlap starts
                    logicol_mtx_trimmed.at[file_b_logicol.index.item(), "collated_end"] = file_a_logicol[
                        "collated_start"].item()

                    # add a new entry for this channel, starting at the where overlap ends
                    j["collated_start"] = file_a_logicol["collated_end"].item()
                    logicol_mtx_trimmed.loc[file_b_logicol.index.item() + 0.5] = j.values[0]
                    logicol_mtx_trimmed = logicol_mtx_trimmed.sort_index()

                    single_excluded_channel(out, file_b_logicol,
                                            f"TRIMMED: {file_a_logicol.index.item()} overlaps entirely with this channel, but not vice-versa."
                                            f"Data is not the same. The overlapping section of this channel appears to only electrical powerline noise, so was trimmed-out.")


                elif (file_a_50hz_proportion > powerline_proportion_threshold) \
                        or (file_a_60hz_proportion > powerline_proportion_threshold):
                    # channel in file A looks to be mostly powerline noise.
                    # file A starts/ends within file B, so can discard, and keep overlapping section from file B.
                    logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=file_a_logicol.index.item())
                    single_excluded_channel(out, file_a_logicol,
                                            f"REMOVED: Overlaps entirely (but not vice-versa, and data is not the same!) with {file_b_logicol.index.item()}, and this appears to only electrical powerline noise, so was removed. ")
                else:
                    # both data appear to be valid...

                    # different data occurs at same time in a and b, but there is more data in b besides the overlapping section, so remove a.
                    # TODO not a perfect solution.
                    logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=file_a_logicol.index.item())
                    single_excluded_channel(out, file_a_logicol,
                                            f"REMOVED: Overlaps entirely (data is not the same!) with {file_b_logicol.index.item()}, but there is data in the other channel besides the overlapping section, so this channel was removed and the other kept.")


            elif overlap["overlap_type"] is OverlapType.ENTIRETY_FILE_B:

                if (file_a_50hz_proportion > powerline_proportion_threshold)  \
                    or (file_a_60hz_proportion > powerline_proportion_threshold):
                    # channel in file A looks to be mostly powerline noise.
                    # file B starts/ends within file A, so trim out the overlapping section from file A, leaving bits on
                    # both sides.

                    j = file_a_logicol.copy()

                    # trim current entry up to the point that the overlap starts
                    logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] = file_b_logicol["collated_start"].item()

                    # add a new entry for this channel, starting at the where overlap ends
                    j["collated_start"] = file_b_logicol["collated_end"].item()
                    logicol_mtx_trimmed.loc[file_a_logicol.index.item() + 0.5] = j.values[0]
                    logicol_mtx_trimmed = logicol_mtx_trimmed.sort_index()

                    single_excluded_channel(out, file_a_logicol,
                                            f"TRIMMED: {file_b_logicol.index.item()} overlaps entirely with this channel, but not vice-versa."
                                            f"Data is not the same. The overlapping section of this channel appears to only electrical powerline noise, so was trimmed-out.")


                elif (file_b_50hz_proportion > powerline_proportion_threshold)  \
                    or (file_b_60hz_proportion > powerline_proportion_threshold):
                    # channel in file B looks to be mostly powerline noise.
                    # file B starts/ends within file A, so can discard, and keep overlapping section from file A.
                    logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=file_b_logicol.index.item())
                    single_excluded_channel(out, file_b_logicol,
                                            f"REMOVED: Overlaps entirely (but not vice-versa, and data is not the same!) with {file_a_logicol.index.item()}, and this appears to only electrical powerline noise, so was removed. ")
                else:
                    # both data appear to be valid...

                    # different data occurs at same time in a and b, but there is more data in a besides the overlapping section, so remove b.
                    # TODO not a perfect solution.
                    logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=file_b_logicol.index.item())
                    single_excluded_channel(out, file_b_logicol,
                                            f"REMOVED: Overlaps entirely (data is not the same!) with {file_a_logicol.index.item()}, but there is data in the other channel besides the overlapping section, so this channel was removed and the other kept.")



            elif overlap["overlap_type"] is OverlapType.ENTIRETY_BOTH_FILES:

                if (file_a_50hz_proportion > powerline_proportion_threshold)  \
                    or (file_a_60hz_proportion > powerline_proportion_threshold):

                    # channel in file A looks to be mostly powerline noise - drop it
                    logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=file_a_logicol.index.item())
                    single_excluded_channel(out, file_a_logicol,
                                            f"REMOVED: Overlaps entirely (and data is not the same!) with {file_b_logicol.index.item()}, and this appears to only electrical powerline noise, so was removed. ")

                elif (file_b_50hz_proportion > powerline_proportion_threshold)  \
                    or (file_b_60hz_proportion > powerline_proportion_threshold):

                    # channel in file B looks to be mostly powerline noise - drop it
                    logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=file_b_logicol.index.item())
                    single_excluded_channel(out, file_b_logicol,
                                            f"REMOVED: Overlaps entirely (and data is not the same!) with {file_a_logicol.index.item()}, and this appears to only electrical powerline noise, so was removed. ")

                else:

                    # remove both channels
                    #logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] = file_a_logicol["collated_start"].item()
                    #logicol_mtx_trimmed.at[file_b_logicol.index.item(), "collated_end"] = file_b_logicol["collated_start"].item()
                    logicol_mtx_trimmed = logicol_mtx_trimmed.drop(index=[file_a_logicol.index.item(), file_b_logicol.index.item()])

                    file_a_logicol_copy = file_a_logicol.copy()
                    file_a_logicol_copy["reason"] = f"*REMOVED: Overlaps entirely with {file_b_logicol.index.item()}, but data is not the same, so both channels removed."
                    file_b_logicol_copy = file_b_logicol.copy()
                    file_b_logicol_copy["reason"] = f"*REMOVED: Overlaps entirely with {file_a_logicol.index.item()}, but data is not the same, so both channels removed."

                    excluded_channels_filename = os.path.join(out, constants.EXCLUDED_CHANNELS_LIST_FILENAME)
                    try:
                        excluded_channels = pd.read_csv(excluded_channels_filename, index_col="index")
                        excluded_channels = pd.concat([excluded_channels, file_a_logicol_copy, file_b_logicol_copy])
                    except FileNotFoundError:
                        excluded_channels = pd.concat([file_a_logicol_copy, file_b_logicol_copy])

                    excluded_channels.to_csv(excluded_channels_filename, index_label="index")
                    print(f"edf_overlaps.resolve: Encountered overlap consisting entirely of both channels involved, but data not the same. Both channels omitted. See *REMOVED in {excluded_channels_filename}. Manual correction recommended.", enabled=constants.VERBOSE)


        # we've fixed an overlap, so re-generate
        overlaps = check(root, out, mtx=logicol_mtx_trimmed, verbose=False)

    # save trimmed logicol_mtx to csv
    logicol_mtx_trimmed.to_csv(os.path.join(out, constants.LOGICOL_POST_OVERLAP_RESOLVE_FILENAME), index_label="index")



