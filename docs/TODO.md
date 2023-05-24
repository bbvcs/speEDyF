
# Overlap Resolution
    # what to do if physically collated file for subject is present?
        # i.e, EVERYTHING overlaps with one file


# Misc
    # 895 - channels g14 vs G14?
    # find the canine data and run it on that
    # standardise output; like how overlap.resolve.check output their names
    # convert all prints to use verbose=constants.VERBOSE, allow this var to be set as param

# Major things left TODO
    # more overlap conditions

    # test whether subject with files of differing sample rate is OK
        # and channels of differing sample rates within same file/segment!

    # take hash of filesystem under root directory
        # save to details.json?
        # so can tell if changed and program won't work anymore
        # TODO; logicol matrix re-generation; not only check for forced, but if hash in details.json has changed.

# Analysis
    # mne-python
