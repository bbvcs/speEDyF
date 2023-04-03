import os
import datetime
import math
import json

import numpy as np
import pandas as pd

import pyedflib

from utils.custom_print import print
from utils import constants


class EDF_Segment:



    def __init__(self, root, out, segment_len_s = 300, cache_lifetime=10):
        """Determine how the logical collation of files under root can be split up into segments of specified length."""

        self.root = root
        self.out = out
        self.segment_len_s = segment_len_s

        # cache used to temporarily store file channel data after reading, as will likely be needed by multiple segments
        self.cache = {}
        self.cache_lifetime = cache_lifetime

        # TODO need to know to load trimmed matrix if needs be - or will we save trim as normal?
        # logicol_mtx holds information on where channels start/end in logical collation, though overlaps may be present
        self.logicol_mtx = pd.read_csv(os.path.join(out, constants.LOGICOL_PRE_OVERLAP_CHECK_FILENAME), index_col="index")

        with open(os.path.join(out, constants.DETAILS_JSON_FILENAME), "r") as details_file:
            details = json.load(details_file)
            self.edf_sample_rate = details["sample_rate"]
            self.edf_channels_superset = details["channels_superset"]

        logicol_start_samples = self.logicol_mtx["collated_start"].iloc[0]
        logicol_end_samples = self.logicol_mtx["collated_end"].iloc[-1]

        # how long will each segment be?
        self.segment_len_samples = self.segment_len_s * self.edf_sample_rate

        # determine segment onset index within logical collation
        segment_onsets = np.arange(logicol_start_samples, logicol_end_samples-self.segment_len_samples, self.segment_len_samples)
        self.segments_mtx = pd.DataFrame({"segment_idx": [i for i in range(0, len(segment_onsets))],
                                          "collated_start": segment_onsets,
                                          "collated_end":   segment_onsets + self.segment_len_samples})


    def get_segment_count(self):
        """How many segments of the specified time can be made?"""
        return self.segments_mtx.shape[0]

    def get_channels(self):
        """Get list of channels present in all files, of which each segment will have a row for"""
        return self.edf_channels_superset

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



    def get_segment(self, idx=None, channels="all", as_DataFrame=True): # TODO collated_start, collated_end param
        """Get a segment via its index."""
        self.cache_lifetime_cycle()

        if channels != "all" and (not isinstance(channels, list) or any([type(ch) != str for ch in channels])):
            raise TypeError("Please ensure 'channels' is either 'all' or a list of strings.")

        # get where the segment starts/ends in whole logical collation mtx
        segment_collated_start = self.segments_mtx[self.segments_mtx["segment_idx"] == idx]["collated_start"].item()
        segment_collated_end = self.segments_mtx[self.segments_mtx["segment_idx"] == idx]["collated_end"].item()

        # find any channels in logicol mtx that overlap (w.r.t index) with segment start/end index
        #segment_channels = self.logicol_mtx.query("not (@segment_collated_end <= collated_start) or (collated_end <= @segment_collated_start)")
        segment_channels = self.logicol_mtx.query(
            "(@segment_collated_end >= collated_start) and (collated_end > @segment_collated_start)")

        # initialise segment
        if channels == "all":
            segment = np.full(shape=(len(self.edf_channels_superset), self.segment_len_samples), fill_value=np.NaN)
        else:
            segment = np.full(shape=(len(channels), self.segment_len_samples), fill_value=np.NaN)

        # for each row of segment_channels matrix
        for file, channel_label, channel_collated_start, channel_collated_end, channel_length \
                in zip(segment_channels["file"], segment_channels["channel"],
                       segment_channels["collated_start"], segment_channels["collated_end"],
                       segment_channels["channel_length"]):

            # get whole channel data
            with pyedflib.EdfReader(file) as edf_file:

                labels = edf_file.getSignalLabels()

                # where is this channel in file, so we can read it
                file_channel_idx = labels.index(channel_label)

                if channels == "all":
                    # where is this channel in agreed-upon channel order in all segments, so we can place in segment correctly
                    segment_channel_idx = self.edf_channels_superset.index(channel_label)
                else:
                    # skip this if channel isn't in specified list
                    if channel_label not in channels:
                        continue
                    segment_channel_idx = channels.index(channel_label)

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
                channel_read_sample_count = min(self.segment_len_samples, channel_collated_end - segment_collated_start)
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
                channel_read_sample_count = channel_length
                channel_read_stop = channel_read_start + channel_read_sample_count

                segment_write_start = channel_collated_start - segment_collated_start
                segment_write_stop = segment_write_start + channel_read_sample_count


            data = channel_data[channel_read_start:channel_read_stop]
            segment[segment_channel_idx, segment_write_start:segment_write_stop] = data

        if as_DataFrame:
            columns = self.edf_channels_superset if channels == "all" else channels
            segment = pd.DataFrame(np.transpose(segment), columns=columns)

        return segment

    def get_segments(self, start_idx=0, end_idx=None, count = None, channels="all", as_DataFrame = True, verbose=True):
        """Get all segments from some segment start index to an end index, or the start index plus a count."""
        segments = []

        if count and end_idx:
            print("Warning: value set for both count and end_idx. count will be ignored.")

        if end_idx == None:
            if count:
                end_idx = start_idx+count
            elif count == None:
                end_idx = self.get_segment_count()

        for i in range(start_idx, end_idx):
            if verbose: print(f"{i}/{end_idx-start_idx}", enabled=True)
            segments.append(self.get_segment(idx=i, channels=channels, as_DataFrame=as_DataFrame))

        self.clear_cache()

        return segments



class EDF_SegmentIterator(EDF_Segment):
    """ Provides an interface to iterate over all segments one-by-one"""

    def __init__(self, root, out, start_idx=0, end_idx=None, count=None, segment_len_s = 300, cache_lifetime=10, channels="all", as_DataFrame=True):
        super().__init__(root, out, segment_len_s = segment_len_s, cache_lifetime=cache_lifetime)

        self._idx = start_idx

        self.channels = channels

        if count and end_idx:
            print("Warning: value set for both count and end_idx. count will be ignored.")

        self.end_idx = end_idx
        if end_idx == None:
            if count:
                self.end_idx = start_idx + count
            elif count == None:
                self.end_idx = self.get_segment_count()


        self.as_DataFrame = as_DataFrame

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < self.end_idx:
            segment = self.get_segment(idx=self._idx, channels=self.channels, as_DataFrame=self.as_DataFrame)
            self._idx += 1
            return segment
        else:
            self.clear_cache()
            raise StopIteration

    def get_channels(self):
        """Get list of channels specified by user if provided, otherwise the
        list of channels present in all files, of which each segment will have a row for"""
        if self.channels == "all":
            return self.edf_channels_superset
        else:
            return self.channels