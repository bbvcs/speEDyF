import datetime
import enum
import os
import itertools

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

class ResolutionMatrixEntry():

    def __init__(self, pair_index: int, channel: str, file_a: str, file_b: str,
                       file_a_start: int,	file_a_end: int, file_a_length: int,
                       file_b_start: int,	file_b_end: int, file_b_length: int,
                       #overlap_start: int,	overlap_end: int,
                       overlap_length: int, overlap_type: OverlapType,
                       action: str, data_loss: int):

        self.pair_index = pair_index     # each pair of overlapping files with have unique index, shared across multiple ResolutionMatrixEntry instances
        self.channel = channel # each common channel has its own ResolutionMatrixEntry
        self.file_a = file_a
        self.file_b = file_b

        self.file_a_start = file_a_start   # when collating, where in the file we shuold read from
        self.file_a_end = file_a_end       # and where to (e.g could be shorter as we omit omit overlapping bit)
        self.file_a_length = file_a_length # number of datapoints in channels in file

        self.file_b_start = file_b_start
        self.file_b_end = file_b_end
        self.file_b_length = file_b_length

        #self.overlap_start = overlap_start
        #self.overlap_end = overlap_end
        self.overlap_length = overlap_length
        self.overlap_type = overlap_type

        #self.common_channels = common_channels
        self.action = action
        self.data_loss = data_loss # if we've had to omit data, how many datapoints have been lost?


    def as_dict(self):
        return {
            "pair_index": self.pair_index,
            "channel": self.channel,
            "file_a": self.file_a,
            "file_b": self.file_b,

            "file_a_start": self.file_a_start,
            "file_a_end": self.file_a_end,
            "file_a_length": self.file_a_length,

            "file_b_start": self.file_b_start,
            "file_b_end": self.file_b_end,
            "file_b_length": self.file_b_length,


            "overlap_length": self.overlap_length,
            "overlap_type": self.overlap_type,

            "action": self.action,
            "data_loss": self.data_loss,
        }

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

"""
def resolve_overlap_OLD(overlapping_pair_count, resolution_mtx_entries: list[ResolutionMatrixEntry], j, hz, channel_check_results,
                        file_a_path, file_a_header,
                        file_b_path, file_b_header,
                        overlap_type: OverlapType, overlap_period):

    common_channels = channel_check_results[0]
    file_a_unique_channels = channel_check_results[1]
    file_b_unique_channels = channel_check_results[2]

    file_a_channel_idxs = {channel:i for i, channel in enumerate(file_a_header["channels"])}
    file_b_channel_idxs = {channel:i for i, channel in enumerate(file_b_header["channels"])}

    overlap_start = overlap_period[0]
    overlap_end = overlap_period[1]
    overlap_duration = overlap_period[2]
    overlap_length = overlap_duration.seconds * hz

    file_a = pyedflib.EdfReader(file_a_path, 0, 1)
    file_b = pyedflib.EdfReader(file_b_path, 0, 1)



    for channel in common_channels:

        match overlap_type:

            case overlap_type.PARTIAL_BOTH_FILES:

                file_a_start = file_a_header["startdate"]
                file_a_end = file_a_header["startdate"] + datetime.timedelta(seconds=file_a_header["Duration"])
                file_a_length = file_a_header["Duration"] * hz

                file_b_start = file_b_header["startdate"]
                file_b_end = file_b_header["startdate"] + datetime.timedelta(seconds=file_b_header["Duration"])
                file_b_length = file_b_header["Duration"] * hz

                # get the idx within EDF rows where overlap begins in each file
                file_a_overlap_start = 0 + abs(file_a_start - overlap_start).seconds * hz
                file_b_overlap_start = 0 + abs(file_b_start - overlap_start).seconds * hz

                # get the overlapping data from both channels
                file_a_data = file_a.readSignal(file_a_channel_idxs[channel], start=file_a_overlap_start, n=overlap_length)
                file_b_data = file_b.readSignal(file_b_channel_idxs[channel], start=file_b_overlap_start, n=overlap_length)

                print("TEMP MAKE ALL CHANELS EQUAL, REMOVE ME!")#, enabled=True)
                file_a_data = np.zeros(shape=file_a_data.shape)
                file_b_data = np.zeros(shape=file_b_data.shape)

                # timevec = pd.date_range(overlap_start, overlap_end, overlap_length)
                # plt.plot(timevec, file_a_data, alpha=0.4)
                # plt.plot(timevec, file_b_data, alpha=0.4)
                # plt.legend([file_a_path, file_b_path])
                # plt.title(f"Overlap Duration (min:sec:ms): {overlap_duration}")
                # plt.ylabel("Physical Value")
                # plt.xlabel("Recording Time\nWARNING: Times are approximate and are for debugging purposes only.")
                # plt.show()
                # plt.clf()
                
                if all(file_a_data == file_b_data):
                    # data is the same in parts of either file, so we can remove the overlapping bit from one file

                    # as it is partial overlap, we know:
                        # neither file is compeltely overlapping
                        #

                    # additionally, as they are sorted prior to this algorithm, we know file_a must start first
                    # (if both start at the same time, but one finishes first, that is NOT PARTIAL)
                    # arbitrarily, keep the overlapping data in a, so read all of A
                    # and remove the overlapping bit from B, so read B from the point where the overlap finishes.

                    #edf_channel_intervals[file_b][channel] =

                    resolution_mtx_entry = ResolutionMatrixEntry(
                        pair_index=overlapping_pair_count,
                        channel=channel,
                        file_a=file_a_path,
                        file_b=file_b_path,

                        file_a_start = 0,
                        file_a_end = file_a_length,
                        file_a_length = file_a_length,

                        file_b_start = 0 + overlap_length,
                        file_b_end = file_b_length,
                        file_b_length = file_b_length,

                        overlap_length=overlap_length,
                        overlap_type=overlap_type,

                        action="Partial and Identical Overlap - Overlapping data removed (omitted) from File B",
                        data_loss=0  # we haven't lost any data, as only omitted data present in another file
                    )

                    resolution_mtx_entries.append(resolution_mtx_entry)


                else:
                    raise NotImplementedError


            case overlap_type.ENTIRETY_FILE_A:
                raise NotImplementedError

            case overlap_type.ENTIRETY_FILE_B:
                raise NotImplementedError

            case overlap_type.ENTIRETY_BOTH_FILES:

                file_a_data = file_a.readSignal(file_a_channel_idxs[channel])
                file_b_data = file_b.readSignal(file_b_channel_idxs[channel])

                if all(file_a_data == file_b_data):

                    # data is the same and the files overlap entirely; we can just completely remove all of one.
                    resolution_mtx_entry = ResolutionMatrixEntry(
                        pair_index = overlapping_pair_count,
                        channel = channel,
                        file_a = file_a_path,
                        file_b = file_b_path,

                        file_a_start = 0,
                        file_a_end  = 0,  # read none of A
                        file_a_length = file_a_header["Duration"] * hz,

                        file_b_start = 0,
                        file_b_end = file_b_header["Duration"] * hz,  # read all of B
                        file_b_length = file_b_header["Duration"] * hz,

                        overlap_length = overlap_length,
                        overlap_type = overlap_type,

                        action = "Exact and Identical Overlap - File A arbitrarily removed (completely omitted)",
                        data_loss = 0  # we haven't lost any data, as only omitted data present in another file
                    )

                    resolution_mtx_entries.append(resolution_mtx_entry)

                else:
                    # now we have problem of which to keep.
                    # firstly, we check the signal content of both as before
                    # maybe one appears to be noise/empty of any content? and other file appears OK? If so, keep other
                    # maybe both appear to be noise/empty, so omit both
                    # if they both appear valid, for now omit both.
                    # TODO

                    raise NotImplementedError
    file_a.close()
    file_b.close()


def resolve_overlap(overlapping_pair_count, resolution_mtx_entries: list[ResolutionMatrixEntry], edf_channel_intervals, hz,
                    channel_label, channel_file_a_interval, channel_file_b_interval,
                    file_a_path, file_a_header,
                    file_b_path, file_b_header,
                    overlap_type: OverlapType, overlap_period):


    file_a_channel_idxs = {channel: i for i, channel in enumerate(file_a_header["channels"])}
    file_b_channel_idxs = {channel: i for i, channel in enumerate(file_b_header["channels"])}

    overlap_start = overlap_period[0]
    overlap_end = overlap_period[1]
    overlap_duration = overlap_period[2]
    overlap_length = overlap_duration.seconds * hz

    # within the lists of intervals for the channel in files a and b, what is index of this interval? (probably 0)
    channel_file_a_interval_idx = edf_channel_intervals[file_a_path][channel_label].index(channel_file_a_interval)
    channel_file_b_interval_idx = edf_channel_intervals[file_b_path][channel_label].index(channel_file_b_interval)

    file_a = pyedflib.EdfReader(file_a_path, 0, 1)
    file_b = pyedflib.EdfReader(file_b_path, 0, 1)

    match overlap_type:

        case overlap_type.PARTIAL_BOTH_FILES:

            file_a_start = file_a_header["startdate"]
            file_a_end = file_a_header["startdate"] + datetime.timedelta(seconds=file_a_header["Duration"])
            file_a_length = file_a_header["Duration"] * hz

            file_b_start = file_b_header["startdate"]
            file_b_end = file_b_header["startdate"] + datetime.timedelta(seconds=file_b_header["Duration"])
            file_b_length = file_b_header["Duration"] * hz

            # get the idx within EDF rows where overlap begins in each file
            file_a_overlap_start = 0 + abs(file_a_start - overlap_start).seconds * hz
            file_b_overlap_start = 0 + abs(file_b_start - overlap_start).seconds * hz

            # get the overlapping data from both channels
            file_a_data = file_a.readSignal(file_a_channel_idxs[channel_label], start=file_a_overlap_start,
                                            n=overlap_length)
            file_b_data = file_b.readSignal(file_b_channel_idxs[channel_label], start=file_b_overlap_start,
                                            n=overlap_length)

            print("TEMP MAKE ALL CHANELS EQUAL, REMOVE ME!")  # , enabled=True)
            file_a_data = np.zeros(shape=file_a_data.shape)
            file_b_data = np.zeros(shape=file_b_data.shape)

            # timevec = pd.date_range(overlap_start, overlap_end, overlap_length)
            # plt.plot(timevec, file_a_data, alpha=0.4)
            # plt.plot(timevec, file_b_data, alpha=0.4)
            # plt.legend([file_a_path, file_b_path])
            # plt.title(f"Overlap Duration (min:sec:ms): {overlap_duration}")
            # plt.ylabel("Physical Value")
            # plt.xlabel("Recording Time\nWARNING: Times are approximate and are for debugging purposes only.")
            # plt.show()
            # plt.clf()

            if all(file_a_data == file_b_data):
                # data is the same in parts of either file, so we can remove the overlapping bit from one file

                # as it is partial overlap, we know:
                    # neither file is compeltely overlapping

                # additionally, as they are sorted prior to this algorithm, we know file_a must start first
                # (if both start at the same time, but one finishes first, that is NOT partial)
                # arbitrarily, keep the overlapping data in a, so read all of A
                # and remove the overlapping bit from B, so read B from the point where the overlap finishes.

                # REPLACE the interval
                edf_channel_intervals[file_b_path][channel_label][channel_file_b_interval_idx] = (overlap_end, channel_file_b_interval[1])


                #
                # resolution_mtx_entry = ResolutionMatrixEntry(
                #     pair_index=overlapping_pair_count,
                #     channel=channel,
                #     file_a=file_a_path,
                #     file_b=file_b_path,
                #
                #     file_a_start=0,
                #     file_a_end=file_a_length,
                #     file_a_length=file_a_length,
                #
                #     file_b_start=0 + overlap_length,
                #     file_b_end=file_b_length,
                #     file_b_length=file_b_length,
                #
                #     overlap_length=overlap_length,
                #     overlap_type=overlap_type,
                #
                #     action="Partial and Identical Overlap - Overlapping data removed (omitted) from File B",
                #     data_loss=0  # we haven't lost any data, as only omitted data present in another file
                # )
                #
                # resolution_mtx_entries.append(resolution_mtx_entry)
                #

            else:
                raise NotImplementedError

        case overlap_type.ENTIRETY_FILE_A:
            raise NotImplementedError

        case overlap_type.ENTIRETY_FILE_B:
            raise NotImplementedError

        case overlap_type.ENTIRETY_BOTH_FILES:

            file_a_data = file_a.readSignal(file_a_channel_idxs[channel_label])
            file_b_data = file_b.readSignal(file_b_channel_idxs[channel_label])

            if all(file_a_data == file_b_data):
            #
            #     # data is the same and the files overlap entirely; we can just completely remove all of one.
            #     resolution_mtx_entry = ResolutionMatrixEntry(
            #         pair_index=overlapping_pair_count,
            #         channel=channel,
            #         file_a=file_a_path,
            #         file_b=file_b_path,
            #
            #         file_a_start=0,
            #         file_a_end=0,  # read none of A
            #         file_a_length=file_a_header["Duration"] * hz,
            #
            #         file_b_start=0,
            #         file_b_end=file_b_header["Duration"] * hz,  # read all of B
            #         file_b_length=file_b_header["Duration"] * hz,
            #
            #         overlap_length=overlap_length,
            #         overlap_type=overlap_type,
            #
            #         action="Exact and Identical Overlap - File A arbitrarily removed (completely omitted)",
            #         data_loss=0  # we haven't lost any data, as only omitted data present in another file
            #     )
            #
            #     resolution_mtx_entries.append(resolution_mtx_entry)
                raise NotImplementedError
            else:
                # now we have problem of which to keep.
                # firstly, we check the signal content of both as before
                # maybe one appears to be noise/empty of any content? and other file appears OK? If so, keep other
                # maybe both appear to be noise/empty, so omit both
                # if they both appear valid, for now omit both.
                # TODO

                raise NotImplementedError
    file_a.close()
    file_b.close()
"""

def interval_plot(interval_list):

    fig, ax = plt.subplots()


    for i in range(0, len(interval_list)):
        start = interval_list[i][0]
        end = interval_list[i][1]
        ax.plot([start, end], [-i, -i])

    return fig, ax



def check(root, out, mtx=None, verbose=False):

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


    logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_RESOLVE_FILENAME), index_col="index")

    # edf_check_overlap will provide an overlap-adjusted mtx we should use instead
    if isinstance(mtx, pd.DataFrame):
        # try to ensure mtx is of correct format
        if list(mtx.columns) == list(logicol_mtx.columns):
            logicol_mtx = mtx
        else:
            raise TypeError("Matrix provided does not appear to be similar to logicol_mtx")

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
            print("No overlaps detected.", enabled=True)
        else:
            n_overlapping_channels = len(pd.unique(overlap_mtx["channel"]))
            n_channels_total = len(pd.unique(logicol_mtx["channel"]))

            n_overlapping_files = overlap_mtx.groupby(["file_A", "file_B"]).ngroups
            n_files_total = len(pd.unique(logicol_mtx["file"]))
            print(f"{n_overlaps} overlapping channels ({n_overlapping_channels}/{n_channels_total} unique channels total) across {n_overlapping_files}/{n_files_total} pairs of files!", enabled=True)
            print("\n", enabled=True)
            print(pd.DataFrame([overlap["overlap_type"] for overlap in overlap_mtx_entries]).value_counts(), enabled=True)

    return overlap_mtx_entries


def resolve(root, out):

    # logicol_mtx holds information on where channels start/end in logical collation, though overlaps may be present
    logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_RESOLVE_FILENAME), index_col="index")

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

        overlaps = check(root, out, mtx=logicol_mtx_trimmed)

        if len(overlaps) == 0:
            break

        # print progress
        if (len(overlaps) >= 100 and len(overlaps) % 100 == 0) \
            or (len(overlaps) >= 10 and len(overlaps) % 10 == 0) \
                or (len(overlaps) < 10):
                    print(f"{len(overlaps)} overlaps left to resolve.", enabled=constants.VERBOSE)

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
        match overlap["overlap_type"]:

            # e.g:
            # fa:   fa_start---------------->fa_end
            # fb:              fb_start------------ ...
            case OverlapType.PARTIAL_BOTH_ENDOF_A:

                file_a_overlap_start = file_b_collated_start - file_a_collated_start
                file_a_overlap_end = file_a_overlap_start + overlap_duration

                file_b_overlap_start = 0
                file_b_overlap_end = file_b_overlap_start + overlap_duration

            case OverlapType.PARTIAL_BOTH_ENDOF_B: # maybe impossible?

                raise NotImplementedError

            case OverlapType.ENTIRETY_FILE_A: # maybe impossible?

                raise NotImplementedError

            case OverlapType.ENTIRETY_FILE_B:

                raise NotImplementedError

            case OverlapType.ENTIRETY_BOTH_FILES:

                raise NotImplementedError


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

        # is overlapping data the same?
        # TODO what about sample rates?
            # surely if differing sample rate, data wont be the same.
            # keep data of higher sample rate?
        if all(file_a_overlap_data == file_b_overlap_data):

            match overlap["overlap_type"]:

                case OverlapType.PARTIAL_BOTH_ENDOF_A:

                    # trim the overlapping data from the end of the channel in file A
                    logicol_mtx_trimmed.at[file_a_logicol.index.item(), "collated_end"] = file_b_logicol["collated_start"].item()

                case OverlapType.PARTIAL_BOTH_ENDOF_B:

                    raise NotImplementedError

                case OverlapType.ENTIRETY_FILE_A:

                    raise NotImplementedError

                case OverlapType.ENTIRETY_FILE_B:

                    raise NotImplementedError

                case OverlapType.ENTIRETY_BOTH_FILES:

                    raise NotImplementedError

        else:  # data isn't the same; more problematic

            match overlap["overlap_type"]:

                case OverlapType.PARTIAL_BOTH_ENDOF_A:

                    raise NotImplementedError

                case OverlapType.PARTIAL_BOTH_ENDOF_B:

                    raise NotImplementedError

                case OverlapType.ENTIRETY_FILE_A:

                    raise NotImplementedError

                case OverlapType.ENTIRETY_FILE_B:

                    raise NotImplementedError

                case OverlapType.ENTIRETY_BOTH_FILES:

                    raise NotImplementedError

    # save trimmed logicol_mtx to csv
    logicol_mtx_trimmed.to_csv(os.path.join(out, constants.LOGICOL_POST_OVERLAP_RESOLVE_FILENAME), index_label="index")



