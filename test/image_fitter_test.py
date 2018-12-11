'''
Created on 20 nov 2018

@author: gcarla
'''
import unittest
import numpy as np
from tesi.image_fitter import ImageFitter
from tesi.image_creator import ImageCreator
from astropy.stats.funcs import gaussian_fwhm_to_sigma


class ImageFitterTest(unittest.TestCase):


    def setUp(self):
        self._threshold= 1000
        self._fwhm= 2
        self._shape=(40,60)
        self._imfit= ImageFitter(self._threshold, self._fwhm)
        self._imageCreator= ImageCreator(self._shape)



    def _testSingleGaussianNoNoise(self,
                                   posX, posY, sigmaX, sigmaY, amplitude):
        ima= self._imageCreator.createGaussianImage(
            posX, posY, sigmaX, sigmaY, amplitude)
        self._imfit.fitSingleStarWithGaussianFit(ima)
        self.assertAlmostEqual(posX, self._imfit.getCentroid()[0])
        self.assertAlmostEqual(posY, self._imfit.getCentroid()[1])
        self.assertAlmostEqual(sigmaX, self._imfit.getSigmaXY()[0])
        self.assertAlmostEqual(sigmaY, self._imfit.getSigmaXY()[1])
        self.assertAlmostEqual(amplitude, self._imfit.getAmplitude())


    def testSingleGaussianNoNoise(self):
        posX= 18.1234
        posY= 22.35
        sigmaX= 1.
        sigmaY= 1.1
        amplitude= 5000
        self._testSingleGaussianNoNoise(
            posX, posY, sigmaX, sigmaY, amplitude)



    
    def testSingleGaussianMultipleTimes(self):
        for i in range(100):
            posX= np.random.uniform(self._fwhm, self._shape[1]-self._fwhm)
            posY= np.random.uniform(self._fwhm, self._shape[0]-self._fwhm)
            sigmaX= np.random.uniform(0.95*self._fwhm, 1.05*self._fwhm
                                      ) * gaussian_fwhm_to_sigma
            sigmaY= np.random.uniform(0.95*self._fwhm, 1.05*self._fwhm
                                      ) * gaussian_fwhm_to_sigma
            amplitude= np.random.uniform(
                2.0*self._threshold, 10.0*self._threshold) 
            try:
                self._testSingleGaussianNoNoise(
                    posX, posY, sigmaX, sigmaY, amplitude)
            except Exception:
                raise Exception(
                    "Iter %d - Could not find star. Params %g %g %g %g %g " %
                    (i, posX, posY, sigmaX, sigmaY, amplitude))
            


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()