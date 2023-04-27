# spe*ED*y*F*
> a speedy EDF (European Data Format - EEG, ECG, etc.) processing pipeline, for time-series analysis of EDF data. 
  
> _Billy C. Smith, CNNP Lab, Newcastle University, UK_

## Features:  

1. Produce a chronological representation of a collection of EDF files under a root directory.
	- **Quick** and **space-efficient**; only requires file headers to do so. No new EDF files are created. 
	- Requires only the root directory, no additional metadata.
	- Accounts for gaps in time between recordings.
	- Check for and deal with **time** (i.e overlapping start/end times) and **space** (i.e EDF channels) **overlaps** between files/channels.
	- Supports files/channels with differing sample rates. (**TODO**)  


2. Break the ordered series into segments of a specified time length (e.g 5 minutes).
	- **Iterable interface** provided to efficiently read segments into memory one-by-one.
	- Supports selection of only specific channels from a segment.
	- Segments formatted as pandas DataFrames, with channel names as headers.  

## Requirements
- Designed with Python 3.8+ in mind. 


## Tutorial
**TODO**
