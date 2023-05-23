import os
import sys
import datetime
import math
import json
import gc

import numpy as np
import pandas as pd

import pyedflib

from .edf_overlaps import check
from .utils.custom_print import print
from .utils import constants


class EDFSegment:
    """Container for segment data, as well as metadata such as sample rate and collation index."""

    def __init__(self, data: pd.DataFrame, sample_rate: float, idx: int = None):

        self.data = data
        self.sample_rate = sample_rate
        self.idx = idx


class EDFSegmenter:

    def __init__(self, root: str, out: str, segment_len_s: int = 300, use_channels="all", cache_lifetime: int = 10):
        """ ...


        Determine how the logical collation of files under root can be split up into segments of specified length.

        """

        self.root = root
        self.out = out
        self.segment_len_s = segment_len_s

        self._iter_idx = 0

        # cache used to temporarily store file channel data after reading, as will likely be needed by multiple segments
        self.cache = {}
        self.cache_lifetime = cache_lifetime

        try:
            self.logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_POST_OVERLAP_RESOLVE_FILENAME), index_col="index")
            print("edf_segment: Overlap-trimmed Logicol Matrix found and successfully loaded", enabled=True)

        except FileNotFoundError:

            if len(check(root, out)) > 0:
                print(f"edf_segment: Warning: Trimmed Logicol Matrix could not be found in {out}, and it appears there are overlaps in your data.\n"
                      f"edf_segment: It is highly recommended that you run edf_overlaps.resolve(), to resolve these overlaps.\n"
                      f"edf_segment: Continue anyway with overlaps present? (y/n)", enabled=True)
                if str(input("> ")).lower() != "y":
                    sys.exit(os.EX_NOINPUT)

            try:
                self.logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_RESOLVE_FILENAME),
                                               index_col="index")

            except FileNotFoundError:
                print(f"edf_segment: Warning: Logicol Matrix could not be found in {out} - have you run edf_collate?", enabled=True)
                sys.exit(os.EX_NOINPUT)

        with open(os.path.join(out, constants.DETAILS_JSON_FILENAME), "r") as details_file:
            details = json.load(details_file)
            self.__available_channels = details["channels_superset"]

        if use_channels != "all" and (not isinstance(use_channels, list) or any([type(ch) != str for ch in use_channels])):
            raise TypeError("edf_segment: Please ensure 'use_channels' is either 'all' or a list of strings.")
        elif use_channels == "all":
            self.use_channels = self.__available_channels.copy()
        else:
            if isinstance(use_channels, list):
                self.use_channels = [ch for ch in use_channels if ch in self.__available_channels]


        logicol_start_s = self.logicol_mtx["collated_start"].iloc[0]
        logicol_end_s = self.logicol_mtx["collated_end"].iloc[-1]

        # determine segment onset index within logical collation
        segment_onsets = np.arange(logicol_start_s, logicol_end_s-self.segment_len_s, self.segment_len_s)
        self.segments_mtx = pd.DataFrame({"segment_idx": [i for i in range(0, len(segment_onsets))],
                                          "collated_start": segment_onsets,
                                          "collated_end":   segment_onsets + self.segment_len_s})

    def get_max_segment_count(self):
        """How many segments of the specified time can be made?"""
        return self.segments_mtx.shape[0]

    def get_available_channels(self):
        """Get list of channels present in the collation, of which each segment will have a row for by default."""
        return self.__available_channels

    def get_used_channels(self):
        """Get the channels present in segments produced by the segmenter - either all available, or a subset."""
        return self.use_channels

    def set_used_channels(self, use_channels):
        """Set the channels used by the segmenter - all segments produced will contain only these channels.

           While you could take a slice of each segment dataframe individually, setting them before producing segments
           saves time and memory - channels you're not interested in won't be loaded.
        """

        if use_channels != "all" and (not isinstance(use_channels, list) or any([type(ch) != str for ch in use_channels])):
            raise TypeError("Please ensure 'use_channels' is either 'all' or a list of strings.")
        elif use_channels == "all":
            self.use_channels = self.__available_channels.copy()
        else:
            if isinstance(use_channels, list):
                self.use_channels = [ch for ch in use_channels if ch in self.__available_channels]

    def clear_cache(self):
        self.cache = {}

    def add_to_cache(self, file, channel_label, channel_data):

        cache_object = {
            "lifetime": self.cache_lifetime,
            "data": channel_data
        }

        if file not in self.cache.keys():
            self.cache[file] = {}

        self.cache[file][channel_label] = cache_object

    def present_in_cache(self, file, channel_label):

        if file not in self.cache.keys():
            return False

        if channel_label not in self.cache[file].keys():
            return False

        return True

    def cache_lifetime_cycle(self):
        """ Reduce lifetime of each item in cache by 1"""

        to_delete = {}

        if self.cache:  # if cache has anything in it
            for file in self.cache:
                for channel in self.cache[file]:
                    self.cache[file][channel]["lifetime"] -= 1

                    if self.cache[file][channel]["lifetime"] <= 0:

                        if file not in to_delete.keys():
                            to_delete[file] = []

                        to_delete[file].append(channel)

        if to_delete:  # if any entries marked for deletion
            for file in to_delete:
                for channel in to_delete[file]:
                    del self.cache[file][channel]
                    gc.collect()

    def get_segment(self, idx: int = None, collated_start: int  = None, collated_end: int = None):
        """ Get a segment via its index in the segments matrix, or a specific collated start & end.

        Via index, the collated_start and collated end in the corresponding row of the segments matrix built in the
        EDFSegmenter constructor is referenced (reccomended). Or, these parameters can be set specifically.

        Collated start and end are values in seconds which determine the start/end of the segment relative to the
        start of the first file in the logical collation (which has collated_start = 0 in logicol_mtx).

        """

        self.cache_lifetime_cycle()

        # are we using index in segment_mtx to determine collated start/end, or have they been provided manually?
        # have to use "is/is not None" as idx, collated start/end are either None or integer types.
        if (idx is None and (collated_start is None and collated_end is None)) \
                or (idx is not None and (collated_start is not None or collated_end is not None)):
            raise TypeError("Either segment index (row in Segment Matrix containing collated start/end) OR a collated start AND end must be provided, but not both. ")

        if idx is not None:
            # get where the segment starts/ends in whole logical collation mtx
            segment_collated_start = self.segments_mtx[self.segments_mtx["segment_idx"] == idx]["collated_start"].item()
            segment_collated_end = self.segments_mtx[self.segments_mtx["segment_idx"] == idx]["collated_end"].item()

        if collated_start is not None and collated_end is not None:
            segment_collated_start = collated_start
            segment_collated_end = collated_end

        # find any channels in logicol mtx that overlap in time with our segment (i.e; channels we will use for segment)
        segment_channels = self.logicol_mtx.query(
            "(@segment_collated_end >= collated_start) and (collated_end > @segment_collated_start)")

        # decide upon a sample rate for this segment, by looking at the channel data we are extracting
        segment_channels_sample_rates = pd.unique(segment_channels["channel_sample_rate"])
        if len(segment_channels_sample_rates) == 0:
            # segment has no files - it is completely a gap in time
            segment_sample_rate = 0  # this causes an empty segment dataframe to be produced

        elif len(segment_channels_sample_rates) > 1:
            raise NotImplementedError("Segments currently only support files of the same sample rate") # TODO
            # remember; if this scenario occurs, AND there is a gap between files,
            # the gap can just be sampled at the decided-upon sample rate for this segment
            # (i.e whatever sample rate we are down/upsampling the other file(s) too)

        else:
            segment_sample_rate = segment_channels_sample_rates[0]  # TEMPORARY

        # (if custom collated start/end specified, segment_len_s may differ from self.segment_len_s)
        segment_len_s = segment_collated_end - segment_collated_start

        segment_len_samples = segment_len_s * segment_sample_rate
        segment_len_samples = int(np.floor(segment_len_samples))

        # initialise segment buffer - where we read relevant channel data in to
        segment_data = np.full(shape=(len(self.use_channels), segment_len_samples), fill_value=np.NaN)

        # for each row of segment_channels matrix
        for file, channel_label, channel_collated_start, channel_collated_end, channel_duration, channel_sample_rate \
                in zip(segment_channels["file"], segment_channels["channel"],
                       segment_channels["collated_start"], segment_channels["collated_end"],
                       segment_channels["channel_duration"], segment_channels["channel_sample_rate"]):

            # get whole channel data
            with pyedflib.EdfReader(file) as edf_file:

                # skip this channel if it isn't in specified list
                if channel_label not in self.use_channels:
                    continue

                # skip this channel if it has been marked for removal via overlap resolution
                if channel_collated_end == channel_collated_start:
                    continue
		

                labels = edf_file.getSignalLabels()
                # where is this channel in file, so we can read it
                file_channel_idx = labels.index(channel_label)

                # where is this channel in agreed-upon channel order in all segments, so we can place in segment correctly
                segment_channel_idx = self.use_channels.index(channel_label)

                # do we already have this channel data cached?
                if self.present_in_cache(file, channel_label):
                    channel_data = self.cache[file][channel_label]["data"]

                    # reset lifetime
                    self.cache[file][channel_label]["lifetime"] = self.cache_lifetime

                else:
                    channel_data = edf_file.readSignal(file_channel_idx)

                    # save to cache
                    self.add_to_cache(file, channel_label, channel_data)


            # where in segment will data be written into?
            segment_write_start = None
            segment_write_stop  = None

            # where in channel will segment data be read from?
            channel_read_start = None
            channel_read_stop  = None

            # if the segment starts the same time as/during a file (most cases)
            # e.g:
            # SEGMENT:          -------------------
            # CHANNEL in FILE:  ----------------------.....
            if segment_collated_start >= channel_collated_start:
                channel_read_start = segment_collated_start - channel_collated_start
                channel_read_sample_count = min(segment_len_s, channel_collated_end - segment_collated_start)
                channel_read_stop = channel_read_start + channel_read_sample_count

                segment_write_start = 0
                segment_write_stop = segment_write_start + channel_read_sample_count


            # if the segment starts before the file in logical collation (rare; start of a file after gap)
            # e.g:
            # SEGMENT:          -------------------
            # CHANNEL in FILE:        ----------------.....
            elif segment_collated_start < channel_collated_start and segment_collated_end <= channel_collated_end:
                channel_read_start = 0
                channel_read_sample_count = segment_collated_end - channel_collated_start
                channel_read_stop = channel_read_start + channel_read_sample_count

                segment_write_start = channel_collated_start - segment_collated_start
                segment_write_stop = segment_write_start + channel_read_sample_count


            # if the segment starts before and ends after the file in logical collation (probably very rare)
            # e.g:
            # SEGMENT:          -------------------
            # CHANNEL in FILE:        -------
            elif segment_collated_start < channel_collated_start and segment_collated_end > channel_collated_end:
                channel_read_start = 0
                channel_read_sample_count = channel_duration
                channel_read_stop = channel_read_start + channel_read_sample_count

                segment_write_start = channel_collated_start - segment_collated_start
                segment_write_stop = segment_write_start + channel_read_sample_count

            # TODO should probably use only one sample rate variable. How to deal with 2 files with differing sample rate?

            channel_read_start  = int(np.floor(channel_read_start * channel_sample_rate))
            channel_read_stop   = int(np.floor(channel_read_stop  * channel_sample_rate))

            segment_write_start = int(np.floor(segment_write_start * segment_sample_rate))
            segment_write_stop  = int(np.floor(segment_write_stop  * segment_sample_rate))

            # read the segment data
            data = channel_data[channel_read_start:channel_read_stop]
            segment_data[segment_channel_idx, segment_write_start:segment_write_stop] = data


        # convert to DataFrame
        columns = self.use_channels
        segment_data = pd.DataFrame(np.transpose(segment_data), columns=columns)

        return EDFSegment(data=segment_data, idx=idx, sample_rate = segment_sample_rate)

    def get_segments(self, start_idx=0, end_idx=None, count=None, verbose=False):
        """Get all segments from some segment start index to an end index, or the start index plus a count."""
        segments = []

        if count and end_idx:
            print("edf_segment: Warning: value set for both count and end_idx. count will be ignored.")

        if end_idx == None:
            if count:
                end_idx = start_idx+count
            elif count == None:
                end_idx = self.get_max_segment_count()

        for i in range(start_idx, end_idx):
            if verbose: print(f"edf_segment: {i-start_idx+1}/{end_idx-start_idx} (idx = {i})", enabled=True)
            segments.append(self.get_segment(idx=i))

        self.clear_cache()

        return segments

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_idx < self.get_max_segment_count():
            segment = self.get_segment(idx=self._iter_idx)
            self._iter_idx += 1
            return segment
        else:
            self.clear_cache()
            self._iter_idx = 0
            raise StopIteration
