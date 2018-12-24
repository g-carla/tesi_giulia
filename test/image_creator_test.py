import unittest
import numpy as np
from tesi.image_creator import ImageCreator, DetectorSpecifications


class ImageCreatorTest(unittest.TestCase):


    def setUp(self):
        self._ic= ImageCreator((40, 40))



    def testNoSourceIdealDetectorMakeNullImage(self):
        ic= ImageCreator((40, 40))
        ic.setDetector(DetectorSpecifications.ideal())
        ic.usePoissonNoise(False)
        ima= ic.createImage()
        self.assertTrue(np.array_equal(np.zeros((40, 40)), ima))


    def testNoSourceDarkCurrentImage(self):
        ic= ImageCreator((40, 40))
        ccd= DetectorSpecifications.ideal()
        ccd.name='onlyDark'
        ccd.darkCurrentInElectronsPerPixelPerSecond= 100.
        ic.setDetector(ccd)
        ic.setExposureTime(1.0)
        ic.usePoissonNoise(False)
        ima= ic.createImage()
        self.assertAlmostEqual(100*1.0, ima.mean(), delta=1)


    def testNoSourceBiasImage(self):
        ic= ImageCreator((40, 40))
        ccd= DetectorSpecifications.ideal()
        ccd.name='onlyBias'
        ccd.biasLevelInAdu= 20.
        ic.setDetector(ccd)
        ic.usePoissonNoise(False)
        ima= ic.createImage()
        self.assertTrue(np.array_equal(np.ones((40, 40))*20, ima))


    def testCreateSingleGaussianImageNoiselessCentered(self):
        sz= 40
        ic= ImageCreator((sz, sz))
        ic.usePoissonNoise(False)
        posX= 20
        posY= 20
        stdX= 2
        stdY= 2
        fluxInPhotons= 1000
        ima= ic.createGaussianImage(posX, posY,
                                    stdX, stdY,
                                    fluxInPhotons)
        self.assertEqual((sz, sz), ima.shape)
        self.assertEqual(ima.max(), ima[posX, posY])
        self.assertEqual(ima[posX+stdX, posY],
                         ima[posX-stdX, posY])
        self.assertAlmostEqual(ima.sum(), fluxInPhotons)


if __name__ == "__main__":
    unittest.main()
