import unittest
import pyabf
import os
from numpy.testing import assert_array_equal

import scarecrow as sc
from scarecrow import __MOCKDATADIR__

multispiker_datafile = os.path.join(__MOCKDATADIR__, "multispiker.abf")

class BasicTests(unittest.TestCase):

    def test_sag_ratio(self):
        abf = pyabf.ABF(multispiker_datafile)
        abf.setSweep(0)
        epoch = 2
        sr_calculated = sc.sag_ratio_abf(abf, epoch)
        sr_known = 0.03198032081127167
        self.assertAlmostEqual(sr_calculated, sr_known)

    def test_rebound_depolarization(self):
        abf = pyabf.ABF(multispiker_datafile)
        abf.setSweep(0)
        epoch = 3
        rd_calculated = sc.rebound_depolarization_abf(abf, epoch)
        rd_known = 11.33626556396484375
        self.assertAlmostEqual(rd_calculated, rd_known)

    def test_Vmin(self):
        abf = pyabf.ABF(multispiker_datafile)
        abf.setSweep(0)
        epoch = 2
        Vmin_calculated = sc.Vmin_abf(abf, epoch)
        Vmin_known = -124.053955078125
        self.assertAlmostEqual(Vmin_calculated, Vmin_known)

    def test_Vrest(self):
        abf = pyabf.ABF(multispiker_datafile)
        abf.setSweep(0)
        epoch = 0
        Vrest_calculated = sc.Vrest_abf(abf, epoch)
        Vrest_known = -77.83406829833984375
        self.assertAlmostEqual(Vrest_calculated, Vrest_known)

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
        pass
        #abf = pyabf.ABF(multispiker_datafile)
        #abf.setSweep(3)
        #p0 = abf.sweepEpochs.p1s[2]
        #p1 = abf.sweepEpochs.p1s[3]
        #t = abf.sweepX[p0:p1]
        #V = abf.sweepY[p0:p1]

        #tind_calculated = get_first_spike_tind(t, V)
        #tind_known = 151
        #self.assertEqual(tind_calculated, tind_known)

    def test_get_all_spike_ind(self):
        pass
        #abf = pyabf.ABF(multispiker_datafile)
        #abf.setSweep(3)
        #p0 = abf.sweepEpochs.p1s[2]
        #p1 = abf.sweepEpochs.p1s[3]
        #t = abf.sweepX[p0:p1]
        #V = abf.sweepY[p0:p1]

        #tind_calculated = sc.get_all_spike_ind(t, V)
        #tind_known = [151, 351, 930, 1456, 1987,
        #              2593, 3209, 3789, 4413]
        #np.assert_array_equal(tind_calculated, tind_known)

    def test_get_avg_spike_frequency(self):
        pass
        #abf = pyabf.ABF(multispiker_datafile)
        #abf.setSweep(3)
        #epoch = 2

        #freq_calculated = sc.get_avg_spike_frequency(abf, epoch)
        #freq_known = 18.770530267480055
        #self.assertEqual(freq_calculated, freq_known)


if __name__ == '__main__':
    unittest.main()
