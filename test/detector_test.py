import unittest
import numpy as np
from tesi.detector import IdealDetector, LuciDetector, \
    BaseDetector


class IdealDetectorTest(unittest.TestCase):


    def testIdealCcdHasEverythingZeros(self):
        ccd= IdealDetector((10, 10))
        self.assertEqual(ccd.biasLevelInAdu, 0)
        self.assertEqual(ccd.ronInElectrons, 0)
        self.assertEqual(ccd.darkCurrentInElectronsPerPixelPerSecond, 0)


class BaseDetectorTest(unittest.TestCase):


    def testBiasImage(self):
        bias=10.0
        ccd= BaseDetector(
            (10, 10),
            "base ccd",
            16,
            0.0,
            1.0,
            0.0,
            1.0,
            bias)

        self.assertTrue(np.array_equal(
            np.ones(ccd.shape)*bias, ccd.biasMapInAdu()))


    def testReadOutNoiseImage(self):
        ron=10.0
        ccd= BaseDetector(
            (100, 100),
            "base ccd",
            16,
            ron,
            1.0,
            0.0,
            1.0,
            0.0)

        ronImage= ccd.readOutNoiseMapInElectrons()
        self.assertAlmostEqual(0, ronImage.mean(), delta=1)
        self.assertAlmostEqual(ron, ronImage.std(), delta=1)



    def testDarkCurrentImage(self):
        dark=100.0
        shape= (100, 100)
        ccd= BaseDetector(
            shape,
            "base ccd",
            16,
            0.0,
            1.0,
            dark,
            1.0,
            0.0)
        expTime=1.0
        ccd.setExposureTime(expTime)
        darkMap= ccd.darkCurrentMapInElectrons()
        self._testDarkMapIsOk(dark, expTime, darkMap)
        expTime=10.0
        ccd.setExposureTime(expTime)
        darkMap= ccd.darkCurrentMapInElectrons()
        self._testDarkMapIsOk(dark, expTime, darkMap)


    def _testDarkMapIsOk(self, dark, expTime, darkMap):
        self.assertAlmostEqual(
            dark*expTime,
            darkMap.mean(),
            delta=3*np.sqrt(dark*expTime/(darkMap.size)))
        self.assertAlmostEqual(
            np.sqrt(dark),
            darkMap.std(),
            delta=3*np.sqrt(dark*expTime))


    def testClipAtSaturation(self):
        bitDepth=8
        ccd= BaseDetector(
            (10, 10),
            "base ccd",
            bitDepth,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0)

        aduImage= np.ones(ccd.shape)*1000
        clippedImage= ccd.clipAtSaturation(aduImage)
        self.assertEqual(255, clippedImage.max())
        self.assertEqual(255, clippedImage.min())


class LuciDetectorTest(unittest.TestCase):

    def testLuciBiasMapHasStripes(self):
        ccd= LuciDetector()
        biasMap= ccd.biasMapInAdu()
        self.assertAlmostEqual(
            2.0* np.mean(biasMap[63, :]),
            np.mean(biasMap[64, :]))


    def testOverrideProperly(self):
        ccd= LuciDetector()
        ccd.ronInElectrons=0
        ccd.setExposureTime(0.)
        ima= ccd.photons2Adu(np.zeros(ccd.shape))
        self.assertAlmostEqual(
            2.0* np.mean(ima[63, :]),
            np.mean(ima[64, :]))


if __name__ == "__main__":
    unittest.main()
