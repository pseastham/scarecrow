class NoSpikeFoundException(Exception):
    """Evoked when trying to find a spike, but none found."""
    pass


class NoMultipleSpikesException(Exception):
    """Evoked when trying to compute spike frequency but less than 2 spikes
    were detected."""
    pass 
