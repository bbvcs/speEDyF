# spe*ED*y*F*
> a speedy EDF (European Data Format - EEG, ECG, etc.) processing pipeline, for time-series analysis of EDF data.
> Billy C. Smith, CNNP Lab, Newcastle University, UK_

## Features: 
> treat a collection of separate EDF files, presumed to represent a long-term recording for one subject, 
> as a continuous timeseries, accounting for gaps in time between recordings and overlaps between files, 
> and provide an interface to analyse variable-length segments of this data. 

1. Produce a chronological representation of a collection of EDF files under a root directory.
	- **Quick** and **space-efficient**; only requires file headers to do so. No new EDF files are created. 
	- Requires only the root directory, no additional metadata.
	- Accounts for gaps in time between recordings.
	- Check for and deal with **time** (i.e overlapping start/end times) and **space** (i.e EDF channels) **overlaps** between files/channels.
	- Supports files/channels with differing sample rates. 

2. Break the ordered series into segments of a specified time length (e.g 5 minutes).
	- **Iterable interface** provided to efficiently read segments into memory one-by-one.
	- Supports selection of only specific channels from a segment.
	- Segments formatted as pandas DataFrames, with channel names as headers.  

## Requirements
- Designed with Python 3.8+ in mind. 


## Tutorial
See example_usage.py.
