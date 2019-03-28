'''
Created on 21 feb 2019

@author: gcarla
'''


import numpy as np
from tesi import image_creator, astrometricError_estimator, ePSF_builder,\
    image_fitter
from cmath import pi
from astropy.stats.funcs import gaussian_sigma_to_fwhm
from astropy.modeling.functional_models import Gaussian2D


class testTheoreticalAstrometricError2():

    def __init__(self,
                 shape=(50, 50)):

        self.shape = shape

    def _createSetOfSameGaussianSource(self,
                                       flux,
                                       stdx,
                                       stdy,
                                       posX=23.77,
                                       posY=17.01
                                       ):
        ima_cre = image_creator.ImageCreator(self.shape)
        ima_cre.usePoissonNoise(True)
        self.imaSet = []
        for i in range(100):
            ima = ima_cre.createGaussianImage(posX, posY, flux, stdx, stdy)
            self.imaSet.append(ima)
        return self.imaSet

#     def _getEPSFOnSingleFrame(self, ima):
#         builder = ePSF_builder.ePSFBuilder(threshold=80,
#                                            fwhm=3.)
#         builder.buildEPSF(ima)
#         epsfModel = builder.getEPSFModel()
#         return epsfModel

    def _fitStarWithGaussianOnSingleFrame(self,
                                          ima,
                                          flux,
                                          sx,
                                          sy,
                                          fitshape,
                                          apertureRadius):
        threshold = flux/(2*pi*sx*sy)
        fwhm = gaussian_sigma_to_fwhm*sx
        gaussPsf = Gaussian2D(amplitude=threshold,
                              x_stddev=sx,
                              y_stddev=sy,
                              x_mean=23.77,
                              y_mean=17.01,
                              theta=0.)
        gaussPsf.fluxname= 'amplitude'
        gaussPsf.xname= 'x_mean'
        gaussPsf.yname= 'y_mean'
        psf_model = gaussPsf
        ima_fit = image_fitter.ImageFitter(thresholdInPhotons=0.1*threshold,
                                           fwhm=fwhm, min_separation=3.,
                                           sharplo=0.1, sharphi=2.0,
                                           roundlo=-1.0, roundhi=1.0,
                                           peakmax=None)
        ima_fit.fitStarsWithBasicPhotometry(image=ima,
                                            model=psf_model,
                                            fitshape=fitshape,
                                            apertureRadius=apertureRadius)
        fitTab = ima_fit.getFitTable()
        return fitTab

    def _fitAllFrames(self, flux, sx, sy):
        fitTabs = []
        imasList = self._createSetOfSameGaussianSource(flux, sx, sy)
        for ima in imasList:
            tab = self._fitStarWithGaussianOnSingleFrame(ima,
                                                         flux,
                                                         sx,
                                                         sy,
                                                         (11, 11),
                                                         10)
            fitTabs.append(tab)
        return fitTabs

    def _estimateAstrometricError(self, flux, sx, sy):
        tabs = self._fitAllFrames(flux, sx, sy)
        estimator = astrometricError_estimator.EstimateAstrometricError(tabs)
        estimator.createCubeOfStarsInfo()
        astromError = estimator.getStandardAstrometricErrorinPixels()
        return astromError
#
#     def _estimateAstrometricError(self, flux, sx, sy, sharplo=0.1, sharphi=2.0,
#                                   roundlo=-1.0, roundhi=1.0, peakmax=5e04):
#         imasList = self._createSetOfSameGaussianSource(flux, sx, sy)
#         threshold = 0.1*flux/(2*pi*sx*sy)
#         fwhm = 0.5*gaussian_sigma_to_fwhm*sx
#
#         est = astrometricError_estimator.EstimateAstrometricError(
#             imasList,
#             threshold,
#             fwhm,
#             sharplo=sharplo,
#             sharphi=sharphi,
#             roundlo=roundlo,
#             roundhi=roundhi,
#             peakmax=peakmax)
#
#         est.fitStarsOnAllFrames(fitshape=(11, 11), apertureRadius=10)
#         est.createCubeOfStarsInfo()
#         astromError = est.getStandardAstrometricErrorinPixels()
#         return astromError

    def estimateAstromErrorForDifferentFluxes(self):
        self.errList = []
        for flux in np.arange(1e03, 1e05, 1e04):
            err = self._estimateAstrometricError(flux, 1.2, 1.2)
            self.errList.append(err)

    def getAstrometricErrorInPixels(self):
        return np.array(self.errList)

#     def getAstrometricErrorInArcsecs(self):
#
#     def plotAstroErrorVsFlux(self):
#
