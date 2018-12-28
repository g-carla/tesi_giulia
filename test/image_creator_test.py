import unittest
import numpy as np
from tesi.image_creator import ImageCreator


class ImageCreatorTest(unittest.TestCase):


    def testDefaultIsNoSourceIdealDetectorNoNoise(self):
        ic= ImageCreator((40, 40))
        ima= ic.createImage()
        self.assertTrue(np.array_equal(np.zeros((40, 40)), ima))


    def testCreateSingleGaussianImageNoiselessCentered(self):
        sz= 40
        ic= ImageCreator((sz, sz))
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



    def testCreateDoubleGaussianImageNoiseless(self):
        sz= 40
        ic= ImageCreator((sz, sz))
        pos1X= 30
        pos1Y= 30
        pos2X= 10
        pos2Y= 15
        stdX= 2
        stdY= 2
        theta= 0
        flux= 1000
        ic.addGaussianSource(pos1X, pos1Y, stdX, stdY, theta, flux)
        ic.addGaussianSource(pos2X, pos2Y, stdX, stdY, theta, flux)
        ima= ic.createImage()
        self.assertEqual((sz, sz), ima.shape)
        self.assertEqual(ima.max(), ima[pos1Y, pos1X])
        self.assertEqual(ima[pos1Y, pos1X], ima[pos2Y, pos2X])
        # new image, must resetTable
        ic.resetTable()
        ic.addGaussianSource(pos1X, pos1Y, stdX, stdY, theta, flux)
        ima= ic.createImage()
        self.assertNotEqual(ima[pos1Y, pos1X], ima[pos2Y, pos2X])



    def testUseRealisticTransmissionMap(self):
        ic= ImageCreator((50, 50))
        ic.setTransmissionMap(ic.TRANSMISSION_MAP_REALISTIC)
        ic.setSkyBackground(ic.SKY_IDEAL)
        ima= ic.createImage()
        # vignetting at corners
        self.assertLess(ima[0, 0], ima[25, 25])
        # same transmission at opposite corners
        self.assertEqual(ima[1, 1], ima[-1, -1])


    def testIdealSkyScalesWithIntegrationTime(self):
        ic= ImageCreator((10, 10))
        ic.setSkyBackground(ic.SKY_IDEAL)
        ic.setExposureTime(1.0)
        ima1= ic.createImage()
        ic.setExposureTime(10.0)
        ima10= ic.createImage()
        self.assertTrue(np.array_equal(ima1*10, ima10))



if __name__ == "__main__":
    unittest.main()
