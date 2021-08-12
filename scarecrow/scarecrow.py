"""
.. module:: scarecrow
   :synopsis: defines functions for computing features for electrophysiology
   data.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelmax

from .exceptions import NoSpikeFoundException, NoMultipleSpikesException


MISSING = NaN

def sag(V):
    """Computes sag using voltage trace."""
    Vmin = np.amin(V)
    Vend = V[-1]
    return Vmin - Vend
def sag_abf(abf, epoch_ind):
    """Computes sag (absolute, not sag ratio) using abf object and epoch index.

    See `sag_ratio` to calculate the sag ratio."""
    p0 = abf.sweepEpochs.p1s[epoch_ind]
    p1 = abf.sweepEpochs.p1s[epoch_ind+1]
    V = abf.sweepY[p0:p1]
    return sag(V)


def sag_ratio(V):
    """Computes sag ratio using voltage trace.

    Sag ratio is computed as

    $$ SR = \frac{V_{min} - V_{end}}{V_{min}} $$"""

    Vmin = np.amin(V)
    Vend = V[-1]
    sr = (Vmin - Vend) / Vmin
    if sr < 0:
        print("Warning: sag ratio being negative indicates there is no sag")
    return sr
def sag_ratio_abf(abf, epoch_ind):
    """Computes sag ratio using abf object and epoch index."""
    p0 = abf.sweepEpochs.p1s[epoch_ind]
    p1 = abf.sweepEpochs.p1s[epoch_ind+1]
    V = abf.sweepY[p0:p1]
    return sag_ratio(V)


def Vmin(V):
    """Computes minimum of voltage."""
    return np.min(V)
def Vmin_abf(abf, epoch_start):
    """Computes minimum Voltage using abf object and epoch index."""
    p0 = abf.sweepEpochs.p1s[epoch_start]
    p1 = abf.sweepEpochs.p1s[epoch_start + 1]

    V = abf.sweepY[p0:p1]
    return Vmin(V)


def Vrest(V):
    """Computes resting membrane potential by taking average of first 30 values

    TODO:
    In the future this should scale average by length of time, not necessary
    amount of data points. e.g. first 10 ms. This will control for different
    frequencies of data collection."""
    return np.average(V[0:30])
def Vrest_abf(abf, epoch_start):
    """Computes resting membrane potential using abf object and epoch index."""
    p0 = abf.sweepEpochs.p1s[epoch_start]
    p1 = abf.sweepEpochs.p1s[epoch_start+1]
    V = abf.sweepY[p0:p1]
    return Vrest(V)


def voltage_drop(V):
    """Computes voltage drop using voltage trace."""
    vmin = Vmin(V)
    resting = Vrest(V)
    return vmin - resting
def voltage_drop_abf(abf, epoch_start):
    """Computes voltage drop using abf object and epoch index."""
    vmin = Vmin_abf(abf, epoch_start)
    resting = Vrest_abf(abf, epoch_start)
    return vmin - resting


def capacitance(tau, Rm):
    """Computes capacitance from the time constant tau and membrane
    resistance Rm."""
    return tau/Rm


def func_exp(x, a, b, c):
    """Return values from a general exponential function. To be used
    for obtaining the time constant."""
    return a * np.exp(b * x) + c


def time_constant(t, V):
    """Computes the time constant (usually called tau) using a fit to an
    exponential function.

    TODO:
    include estimate for 'fitness' e.g. if the t/V curve doesn't really
    look like an exponential, an error or exception would be nice to throw
    so that junk doesn't come out."""
    a_init = 1
    b_init = -100
    c_init = V[-1]

    popt, pcov = curve_fit(func_exp, t, V, p0=[a_init, b_init, c_init],
                           bounds=(-np.inf, np.inf))

    Vpred = np.zeros(len(t))
    for i in range(len(t)):
        Vpred[i] = func_exp(t[i], popt[0], popt[1], popt[2])

    return -1/popt[1]
def time_constant_abf(abf, epoch_start):
    """Computes time constant using abf object and epoch index."""
    p0 = abf.sweepEpochs.p1s[epoch_start]
    p1 = abf.sweepEpochs.p1s[epoch_start + 1]

    t = abf.sweepX[p0:p1] - abf.sweepX[p0]
    V = abf.sweepY[p0:p1]

    return time_constant(t, V)


def input_membrane_resistance(I, V):
    """Computes input membrane resistance Rm using current trace I and
    voltage trace V."""
    V1 = V[0]
    V2 = V[-1]
    I1 = I[0]
    I2 = I[-1]

    dV = V2 - V1
    dI = I2 - I1

    return dV / dI
def input_membrane_resistance_abf(abf, epoch_start):
    """Computes input membrane resistance Rm using abf object and epoch
    index."""
    p0 = abf.sweepEpochs.p1s[epoch_start]
    p1 = abf.sweepEpochs.p1s[epoch_start + 1]

    V = abf.sweepY[p0:p1]
    I = abf.sweepC[p0-1:p1]

    return input_membrane_resistance(I, V)


def rebound_depolarization_abf(abf, epoch_start):
    Vr = Vrest_abf(abf, 0)

    p0 = abf.sweepEpochs.p1s[epoch_start]

    V = abf.sweepY[p0:-1]
    Vmax = np.max(V)

    return Vmax - Vr


def spike_amplitude(V, t_spike):
    """Computes spike amplitude from voltage trace V and
    spike index t_spike."""
    # handle no spike found
    if t_spike is None:
        return None
    Vmax = V[t_spike]
    Vmin = np.min(V[t_spike+1:t_spike+500])

    return Vmax - Vmin
def spike_amplitude_abf(abf, t_spike, epoch_start=3):
    """Computes spike amplitude from abf object with epoch index and the
    index of the spike time.

    Note that t_spike should be found within the same epoch, otherwise there
    be an index mismatch."""
    # handle no spike found
    if t_spike is None:
        return None
    p0 = abf.sweepEpochs.p1s[epoch_start]
    V = abf.sweepY[p0:-1]

    return spike_amplitude(V, t_spike)


def find_nearest_idx(arr, val):
    """Finds index in an array `arr` closest to value `val`."""
    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return idx


def spike_width(t, V, t_spike, spike_amp):
    """Computes spike width for time t, voltage trace V, and index t_spike
    and voltage amplitude `spike_amp`."""
    # handle no spike found
    if t_spike is None:
        return None

    Vmin = np.min(V[t_spike+1:t_spike+500])
    id1 = find_nearest_idx(V[t_spike-100:t_spike], spike_amp/2 + Vmin) \
        + t_spike - 100
    id2 = find_nearest_idx(V[t_spike+1:t_spike+500], spike_amp/2 + Vmin) \
        + t_spike + 1
    return t[id2] - t[id1]
def spike_width_abf(abf, t_spike, spike_amp, epoch_start=3):
    """Computes spike width for abf object and t_spike index, and spike
    amplitude `spike_amp`.

    Note that t_spike should be found within the same epoch, otherwise there
    be an index mismatch."""
    # handle no spike found
    if t_spike is None:
        return None
    p0 = abf.sweepEpochs.p1s[epoch_start]
    t = abf.sweepX[p0:-1]
    V = abf.sweepY[p0:-1]
    return spike_width(t, V, t_spike, spike_amp)


def first_spike_tind(V, startind=0):
    """Finds the index of the first spike. The value of startind can be
    used as an offset in case t and V are slices of a larger array, but you
    want the index for those arrays."""
    tarr = argrelmax(V, order=5)[0]
    found_spike = False

    for val in tarr:
        if (V[val] > -20) and val > 0:      # val > 0 to avoid repeats
            spike_tind = val
            found_spike = True
            break

    if found_spike is False:
        raise NoSpikeFoundException
    else:
        return spike_tind + startind
def first_spike_tind_abf(abf, epoch_start, startind=0):
    """ returns t_max for spike time """
    p0 = abf.sweepEpochs.p1s[epoch_start]
    V = abf.sweepY[p0:-1]
    return first_spike_tind(V, startind=startind)


def spike_latency(t, I, V):
    """Computes spike latency.

    Makes sure that current is +100 pA.
    """
    # make sure that current is +100 pA
    if abs(I[5] - 0.1) > 1e-7:
        sign = ""
        if I[5] > 0:
            sign = "+"
        print(f"Warning! Expected +100pA current, got {sign}{round(I[5]*1000)} \
                pA current")

    spike_tind = first_spike_tind(V)
    return t[spike_tind] - t[0]
def spike_latency_abf(abf, epochstart):
    """Computes spike latency using abf objet and epoch index."""
    p0 = abf.sweepEpochs.p1s[epochstart]
    t = abf.sweepX[p0:-1]
    V = abf.sweepY[p0:-1]
    I = abf.sweepC[p0:-1]
    return spike_latency(t, I, V)


def all_spike_ind(t, V):
    """Gets all spike indices from time t and voltage trace V."""
    spike_indices = []
    indval = 0
    while indval < len(t):
        try:
            tspike = first_spike_tind(V[indval:], startind=indval)
            spike_indices.append(tspike)
            indval = tspike
        except NoSpikeFoundException:
            indval = len(t) + 1

    return spike_indices


def interspike_intervals(t, V):
    """Computes interspike intervals for time t and voltage trace V.

    If there are N spikes, then there will be N-1 intervals."""
    # first pass -- get number of spikes and locations
    spike_inds = all_spike_ind(t, V)
    n_spikes = len(spike_inds)

    if n_spikes == 0:
        return []

    # generate array to hold time intervals
    intervals = np.zeros((n_spikes-1), dtype=float)
    for ti in range(1, n_spikes):
        intervals[ti-1] = t[spike_inds[ti]] - t[spike_inds[ti-1]]

    return intervals


def avg_spike_frequency(t, V):
    """Computes inter-spike intervals for each spike, then
    computes the average of those intervals, then returns the
    reciprocal to denote the average spike frequency, in Hz.

    Note: This is in contrast to simply counting the number of
    total spikes and dividing by the time of applied current.
    In general, this method should give a more accurate value,
    especially in the case of a neuron experiencing depolarization
    block.

    Note: If there are zero or one spikes, then this function returns 0.
    """
    intervals = interspike_intervals(t, V)

    try:
        raise_if_not_multiple_spikes(intervals)
    except NoMultipleSpikesException:
        return None

    avg_int = np.average(intervals)
    return 1/avg_int
def avg_spike_frequency_abf(abf, epoch):
    """Computes average spike frequency for abf object and epoch index."""
    p0 = abf.sweepEpochs.p1s[epoch]
    p1 = abf.sweepEpochs.p1s[epoch+1]
    t = abf.sweepX[p0:p1]
    V = abf.sweepY[p0:p1]
    return avg_spike_frequency(t, V)


def max_spike_frequency(t, V):
    """Computes maximum inter-spike frequency (equivalent to
    minimum interspike interval).
    """
    intervals = interspike_intervals(t, V)
    raise_if_not_multiple_spikes(intervals)
    min_int = np.amin(intervals)
    return 1/min_int


def min_spike_frequency_tV(t, V):
    """Computes minimum inter-spike frequency (equivalent to
    maximum interspike interval).
    """
    intervals = interspike_intervals(t, V)
    raise_if_not_multiple_spikes(intervals)
    max_int = np.amax(intervals)
    return 1/max_int


def get_spike_frequency_adaptation(t, V):
    """Computes spike adaptation ratio.

    Spike adaptation ratio is defined to be the ratio of the last over
    the first interspike interval.

    "An AR of 1 indicates there is no adaptation, whereas a larger AR indicates
    a slowing of spiking over the current injection" from Ross et al (2019)
    "Experience-Dependent Intrinsic Plasticity During Auditory Learning"
    Journal of Neuroscience.
    """
    # check that there are 2 spikes minimum
    intervals = interspike_intervals(t, V)
    raise_if_not_multiple_spikes(intervals)
    return intervals[-1]/intervals[0]


def raise_if_not_multiple_spikes(intervals):
    """Checks for whether there are multiple spikes, otherwise raises
    and exception."""
    if len(intervals) < 1:
        raise NoMultipleSpikesException


# delete these in production - this is just for development
#import matplotlib.pyplot as plt


#def plot_full(abf):
#    fig = plt.figure(figsize=(8, 5))
#
#    ax1 = fig.add_subplot(211)
#    for sweepNumber in abf.sweepList:
#        abf.setSweep(sweepNumber)
#        ax1.plot(abf.sweepY, color='b')
#        ax1.set_ylabel("ADC (measurement)")
#        ax1.set_xlabel("sweep point (index)")

#    ax2 = fig.add_subplot(212)
#    for sweepNumber in abf.sweepList:
#        abf.setSweep(sweepNumber)
#        ax2.plot(abf.sweepC, color='r')
#        ax2.set_ylabel("DAC (command)")
#        ax2.set_xlabel("sweep point (index)")

#    for p1 in abf.sweepEpochs.p1s:
#        ax1.axvline(p1, color='k', ls='--', alpha=.5)
#        ax2.axvline(p1, color='k', ls='--', alpha=.5)

#    plt.tight_layout()
#    plt.show()