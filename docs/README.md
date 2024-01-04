# spe*ED*y*F*
> a speedy EDF (European Data Format - EEG, ECG, etc.) processing pipeline, for time-series analysis of EDF data.  
> Billy C. Smith, CNNP Lab, Newcastle University, UK

## Features: 
> Treat a collection of separate EDF files, presumed to represent a long-term recording for one subject, 
> as a continuous timeseries, accounting for gaps in time between recordings and overlaps between files, 
> and provide an interface to analyse variable-length segments of this data. 

1. Produce a chronological representation of a collection of EDF files under a root directory.
	- **Quick** and **space-efficient**; only requires file headers to do so. No new EDF files are created. 
	- Requires only the root directory, no additional metadata.
	- Accounts for gaps in time between recordings.
	- Check for and deal with **time** (i.e overlapping start/end times) and **space** (i.e EDF channels) **overlaps** between files/channels.
	- Supports collation of files/channels with differing sample rates.
      - TODO: Channels with different sample rates supported in collation but not yet overlap resolution - coming soon.


2. Break the ordered series into segments of a specified time length (e.g 5 minutes).
	- **Iterable interface** provided to efficiently read segments into memory one-by-one.
	- Supports selection of only specific channels from a segment.
    - Supports changing channel sample rate through recording.
	- Segments formatted as pandas DataFrames, with channel names as headers. 
    - **Currently, only one sample rate is supported across all channels per segment!**
      - All other channels in the segment are downsampled to the minimum.
      - A work-around is to group channels by sample rate, and use the channel-specification feature mentioned above to extract the segment for each group separately.

## Assumptions:
This program was designed to be ran using input data from a root directory containing many EDF
files for a single subject/patient, that represent a split-up continuous recording. This might look something like: 
'/edf_subjects/subject1'. The output of the program is saved to a specified output directory for this
subject, e.g '/my_speedyf_out/subject1'.

## Summary:	 
- edf_collate: Produces a 'logicol_mtx'; a matrix containing the chronological order of files and their channels in the root directory.
- edf_overlaps:
  - .check(): looks for any overlapping data in the logicol matrix, which are ... TODO
  - .resolve(): fixes these overlaps by adjusting the logicol_mtx (logicol_mtxTRIMMED)
- edf_segment: Given the output directory, use the logicol_mtx to read in 'segments' of the continuous data from the original EDF files.

## Requirements
- Designed to work with Python 3.8 and above.
- numpy, matplotlib, pandas, scipy, **pyedflib**

## Tutorial
See example_usage.py.



### Tips
- You only have to run collation and overlap resolution once, though you should run them again if data in the subject's directory changes (e.g you add new EDF data)
  - edf_collate and edf_overlaps.resolve will stop prematurely if it looks like the output files are already there.
    - However, they will also check themselves if it looks like the data has changed, and re-run if so.
- So if you have already run collation/overlap resolution, you can jump straight to edf_segment.

- WARNING: **IF, AFTER RUNNING, YOU MOVE THE LOCATION OF THE INPUT EDF DATA ON DISK**:
  - e.g if you run speEDyF on /old_place/edf_subjects/subject1, then move subject1 to /new_place/edf_subjects/subject1:
  - edf_collate and overlap_resolve may flag that the data has changed and re-run.
  - You can still run edf_segment, but you **must** pass a value for the new_root parameter.
    - e.g instead of running edf_segment.EDFSegmenter(/speedyf_out/subject1), run:
    - edf_segment.EDFSegmenter(/speedyf_out/subject1, new_root = /new_place/edf_subjects/subject1)
      - This will also change the "root" value in the details.json file produced by speEDyF for this subject.