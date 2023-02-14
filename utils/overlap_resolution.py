import datetime
import enum
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyedflib


class OverlapType(enum.Enum):

    NO_OVERLAP = 1,

    # e.g:
    # f1:   t1_start---------------->t1_end
    # f2:              t2_start------------ ...
    PARTIAL_BOTH_FILES = 2,

    # e.g:
    # f1:           t1_start-------->t1_end
    # f2:  t2_start--------------------------->t2_end
    ENTIRETY_FILE_A = 3,  # the situation above
    ENTIRETY_FILE_B = 4,  # the situation above but flipped

    # e.g:
    # f1:  t1_start----------------->t1_end
    # f2:  t2_start----------------->t2_end
    ENTIRETY_BOTH_FILES = 5

class ResolutionMatrixEntry():

    def __init__(self, index: int, channel: str, file_a: str, file_b: str,
                       file_a_start: int,	file_a_end: int, file_a_length: int,
                       file_b_start: int,	file_b_end: int, file_b_length: int,
                       #overlap_start: int,	overlap_end: int,
                       overlap_length: int, overlap_type: OverlapType,
                       action: str, data_loss: int):

        self.index = index     # each pair of overlapping files with have unique index, shared across multiple ResolutionMatrixEntry instances
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
            "index": self.index,
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
    Return Overlap type and overlap start/end time if two recordings overlap in time, otherwise No Overlap and None.
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

            type = OverlapType.PARTIAL_BOTH_FILES
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

            type = OverlapType.PARTIAL_BOTH_FILES
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

def resolve_overlap(overlapping_pair_count, resolution_mtx_entries: list[ResolutionMatrixEntry], hz, channel_check_results,
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


                    resolution_mtx_entry = ResolutionMatrixEntry(
                        index=overlapping_pair_count,
                        channel=channel,
                        file_a=file_a_path,
                        file_b=file_a_path,

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
                        index = overlapping_pair_count,
                        channel = channel,
                        file_a = file_a_path,
                        file_b = file_a_path,

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


