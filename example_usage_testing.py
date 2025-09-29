import os
import time
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))
from speedyf import edf_collate, edf_overlaps, edf_segment


#SUBJECT_DATA_ROOT = "/media/b9038224/Hodur/DATA/UCLH78_EDFs/{subject}"
SUBJECT_DATA_ROOT = "/media/b9038224/Elements/HODUR_BACKUP/UCLH78_EDFs/{subject}"
SUBJECT_DATA_OUT = "/home/campus.ncl.ac.uk/b9038224/University/PhD/Projects/speedyf/out/{subject}"




if __name__ == "__main__":

    start = time.time()

    subject = "931"
    root = SUBJECT_DATA_ROOT.format(subject=subject)
    out = SUBJECT_DATA_OUT.format(subject=subject)

    # produce collation matrix (saved as .csv to out)
    edf_collate(root, out)

    # *check* for overlaps (quick compared to resolving)
    if len(edf_overlaps.check(out, verbose=True)) != 0:
        edf_overlaps.resolve(root, out) # fix them if so (trim matrix) (can take a while)



    # set up segmenter object, to break data into 5 min segments
    segmenter = edf_segment.EDFSegmenter(out, segment_len_s=300)



    segmenter.set_used_channels(["ECG"])
    whole_recording_buffer, metadata = segmenter.whole_recording(out, regenerate=True, verbose=True)



    # maybe we want to use only EEG channels
    segmenter.set_used_channels([ch for ch in segmenter.get_available_channels() if ch != "ECG"])
    # or use all
    segmenter.set_used_channels(segmenter.get_available_channels())

    # get a specific segment
    #segment = segmenter.get_segment(idx=34)

    # get some specific segments
    #segments = segmenter.get_segments(start_idx=200, count=20, verbose=True)

    #segmenter._iter_idx = 3471

    # iterate over all segments in a memory-efficient way, starting from the first
    for segment in segmenter:
        print(f"{segment.idx+1}/{segmenter.get_max_segment_count()}")

        print(segment.get_segment_startdate())

        # segment data in pandas DataFrame format
        data = segment.data

        # get a specific channel
        #channel = segmenter.get_used_channels()[0]
        #channel_data = data[channel]

        # TODO cleanup bandpower script, commit script dir, add example of usage


    end = time.time()
    print(f"Time elapsed: {end-start}")
