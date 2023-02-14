import numpy as np


def header_indicated_sample_rate(edf_header):
    """
    Return the sample rate for an EDF file indicated by signal headers.
    If there are multiple sample rates, pick the most popular
    """

    signal_header_sample_rates = np.array([signal_header["sample_rate"] for signal_header in edf_header["SignalHeaders"]], dtype = np.int16)

    unique_sample_rates, frequency = np.unique(signal_header_sample_rates, return_counts=True)

    # in most cases, all will be the same.
    if len(unique_sample_rates) == 1:
        return unique_sample_rates[0]
    else:
        # tommyp's answer, https://stackoverflow.com/questions/51737245/how-to-sort-a-numpy-array-by-frequency
        sorted_indexes = np.argsort(frequency)[::-1]
        sorted_by_freq = unique_sample_rates[sorted_indexes]
        return sorted_by_freq[0] # return the most frequent value


def all_header_indicated_sample_rates(edf_headers):
    """Return a list of all the unique header-indicated sample rates across all files, ordered by frequency"""

    edf_sample_rates = {}
    for file, header in edf_headers.items():
        edf_sample_rates[file] = header_indicated_sample_rate(header)

    unique_sample_rates, frequency = np.unique(list(edf_sample_rates.values()), return_counts=True)
    sorted_indexes = np.argsort(frequency)[::-1]
    sorted_by_freq = unique_sample_rates[sorted_indexes]

    return sorted_by_freq