import time
import sys

sys.path.append("/home/bcsm/University/stage-4/MSc_Project/Scripts")
from speedyf import edf_collate, edf_overlaps, edf_segment


if __name__ == "__main__":

    start = time.time()

    # TODO: add argparse library, pass in root/out?
    root = "/home/bcsm/University/stage-3/BSc_Project/program/code/FILES/INPUT_DATA/909"

    # dir to save results for this run i.e matrix, overlap resolutions
    out = "out/testing"

    # subject = "909"
    # root = "/home/.../data/{subject}"
    # out = "out/{subject}"

    edf_collate(root, out)

    if len(edf_overlaps.check(root, out, verbose=True)) != 0:
        edf_overlaps.resolve(root, out)

    # set up segmenter object, to break data into 5 min segments
    segmenter = edf_segment.EDFSegmenter(root, out, segment_len_s=300)

    # maybe we want to use only EEG channels
    segmenter.set_used_channels([ch for ch in segmenter.get_available_channels() if ch != "ECG"])
    # maybe we change our mind
    segmenter.set_used_channels(segmenter.get_available_channels())

    # get a specific segment
    segment = segmenter.get_segment(idx=200)

    # get some specific segments
    segments = segmenter.get_segments(start_idx=200, count=20, verbose=True)

    # iterate over all segments, starting from the first
    for segment in segmenter:
        print(f"{segment.idx+1}/{segmenter.get_max_segment_count()}")

        # get a specific channel
        channel = segmenter.get_used_channels()[0]
        channel_data = segment.data[channel]


    end = time.time()
    print(f"Time elapsed: {end-start}")