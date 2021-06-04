import unittest
import scarecrow.postprocessing as pop
from scarecrow import __MOCKDATADIR__
import pyabf
import numpy.testing as np
import os

multispiker_datafile = os.path.join(__MOCKDATADIR__, "multispiker.abf")

class postprocessingTest(unittest.TestCase):

    def test_get_sag_ratio(self):
        pass

    def test_get_rebound_depolarization(self):
        pass

    def test_get_Vmin(self):
        pass

    def test_get_resting(self):
        pass

    def test_get_voltage_drop(self):
        pass

    def test_get_capacitance(self):
        pass

    def test_func_exp(self):
        pass

    def test_get_time_constant(self):
        pass

    def test_get_input_membrane_resistance(self):
        pass

    def test_compute_rebound_depolarization(self):
        pass

    def test_find_first_spike_tind(self):
        pass

    def test_get_spike_amplitude(self):
        pass

    def test_find_nearest_idx(self):
        pass

    def test_get_spike_width(self):
        pass

    def test_get_spike_latency(self):
        pass

    def test_get_first_spike_tind(self):
        abf = pyabf.ABF(multispiker_datafile)
        abf.setSweep(3)
        p0 = abf.sweepEpochs.p1s[2]
        p1 = abf.sweepEpochs.p1s[3]
        t = abf.sweepX[p0:p1]
        V = abf.sweepY[p0:p1]

        tind_calculated = pop.get_first_spike_tind(t, V)
        tind_known = 151
        self.assertEqual(tind_calculated, tind_known)

    def test_get_all_spike_ind(self):
        abf = pyabf.ABF(multispiker_datafile)
        abf.setSweep(3)
        p0 = abf.sweepEpochs.p1s[2]
        p1 = abf.sweepEpochs.p1s[3]
        t = abf.sweepX[p0:p1]
        V = abf.sweepY[p0:p1]

        tind_calculated = pop.get_all_spike_ind(t, V)
        tind_known = [151, 351, 930, 1456, 1987,
                      2593, 3209, 3789, 4413]
        np.assert_array_equal(tind_calculated, tind_known)

    def test_get_avg_spike_frequency(self):
        abf = pyabf.ABF(multispiker_datafile)
        abf.setSweep(3)
        epoch = 2

        freq_calculated = pop.get_avg_spike_frequency(abf, epoch)
        freq_known = 18.770530267480055
        self.assertEqual(freq_calculated, freq_known)


if __name__ == '__main__':
    unittest.main()
