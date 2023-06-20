import os
import time
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))
from speEDyF import edf_collate, edf_overlaps, edf_segment


if __name__ == "__main__":

    start = time.time()

    subject = "1379_ECGEKG"
    root = f"/home/bcsm/University/stage-4/MSc_Project/UCLH/{subject}"
    out = f"out/{subject}"

    # produce collation matrix (saved as .csv to out)
    edf_collate(root, out)

    # *check* for overlaps (quick compared to resolving)
    if len(edf_overlaps.check(root, out, verbose=True)) != 0:
        edf_overlaps.resolve(root, out) # fix them if so (trim matrix) (can take a while)

    # set up segmenter object, to break data into 5 min segments
    segmenter = edf_segment.EDFSegmenter(root, out, segment_len_s=300)

    # maybe we want to use only EEG channels
    segmenter.set_used_channels([ch for ch in segmenter.get_available_channels() if ch != "ECG"])
    # or use all
    segmenter.set_used_channels(segmenter.get_available_channels())

    # get a specific segment
    #segment = segmenter.get_segment(idx=34)

    # get some specific segments
    #segments = segmenter.get_segments(start_idx=200, count=20, verbose=True)

    # iterate over all segments in a memory-efficient way, starting from the first
    for segment in segmenter:
        print(f"{segment.idx+1}/{segmenter.get_max_segment_count()}")

        print(segment.get_segment_startdate())

        # segment data in pandas DataFrame format
        data = segment.data

        # get a specific channel
        channel = segmenter.get_used_channels()[0]
        channel_data = data[channel]

        # TODO cleanup bandpower script, commit script dir, add example of usage


    end = time.time()
    print(f"Time elapsed: {end-start}")
