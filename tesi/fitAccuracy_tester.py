'''
Created on 14 feb 2019

@author: gcarla
'''
import numpy as np
from tesi import image_creator, ePSF_builder, image_fitter
from cmath import pi
from astropy.stats.funcs import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma


class testFitAccuracy():

    def __init__(self,
                 shape=(50, 50),
                 posX=23.77,
                 posY=17.01,
                 fwhm=1):
        self._shape = shape
        self._posX = posX
        self._posY = posY
        self._fwhm = fwhm
        self._stdX = gaussian_fwhm_to_sigma*fwhm
        self._stdY = gaussian_fwhm_to_sigma*fwhm

    def _createSetOfSameGaussianSource(self,
                                       flux,
                                       howManyImages=100):
        ima_cre = image_creator.ImageCreator(self._shape)
        ima_cre.usePoissonNoise(True)
        self.imaSet = []
        print("Create %d images" % howManyImages)
        for i in range(howManyImages):
            ima = ima_cre.createGaussianImage(self._posX, self._posY,
                                              flux, self._stdX, self._stdY)
            self.imaSet.append(np.ma.masked_array(ima))
        return self.imaSet

#     def _getEPSFforSetOfSameGaussian(self, flux):
#         epsfList = []
#         for ima in self._createSetOfSameGaussianSource(flux):
#             builder = ePSF_builder.ePSFBuilder(ima, threshold=800, fwhm=3.,
#                                                size=20, peakmax=None)
#             builder.extractStars()
#             builder.buildEPSF()
#             epsfModel = builder.getEPSFModel()
#             epsfList.append(epsfModel)
#         return epsfList
#
#     def _fitStarsInSetOfSameGaussian(self, flux):
#         threshold = 0.1*flux/(2*pi*self._stdX*self._stdY)
#         fwhm = 0.7*gaussian_sigma_to_fwhm*self._stdX
#         imas = self._createSetOfSameGaussianSource(flux)
#         epsfs = self._getEPSFforSetOfSameGaussian(flux)
#         ima_fit = image_fitter.ImageFitter(thresholdInPhotons=threshold,
#                                            fwhm=fwhm, min_separation=3.,
#                                            sharplo=0.1, sharphi=2.0,
#                                            roundlo=-1.0, roundhi=1.0,
#                                            peakmax=None)
#         for i in range(len(imas)):
#             ima_fit.fitStarsWithBasicPhotometry(image=imas[i],
#                                                 model=epsfs[i],
#                                                 fitshape=(11,11),
#                                                 apertureRadius=10)
#             fitTab = ima_fit.getFitTable()
#             self._posX - fitTab['x_fit']

    def _getEPSFOnSingleFrame(self, ima):
        print("Building EPSF")
        builder = ePSF_builder.ePSFBuilder(ima, threshold=80, fwhm=self._fwhm,
                                           size=20, peakmax=None)
        builder.extractStars()
        builder.buildEPSF()
        epsfModel = builder.getEPSFModel()
        return epsfModel

    def _fitStarWithEPSFOnSingleFrame(self,
                                      ima,
                                      flux,
                                      epsfModel,
                                      fitshape,
                                      apertureRadius):
        print("Fitting star")
        threshold = 0.1*flux/(2*pi*self._stdX*self._stdY)
        fwhm = 0.7*self._fwhm
        ima_fit = image_fitter.ImageFitter(thresholdInPhotons=threshold,
                                           fwhm=fwhm, min_separation=3.,
                                           sharplo=0.1, sharphi=2.0,
                                           roundlo=-1.0, roundhi=1.0,
                                           peakmax=None)
        ima_fit.fitStarsWithBasicPhotometry(image=ima,
                                            model=epsfModel,
                                            fitshape=fitshape,
                                            apertureRadius=apertureRadius)
        fitTab = ima_fit.getFitTable()
        return fitTab

    def _fitAllFrames(self, flux, model):
        self._fitTabs = []
        imasList = self._createSetOfSameGaussianSource(flux)
        print("Fitting images")
        for ima in imasList:
            tab = self._fitStarWithEPSFOnSingleFrame(ima,
                                                     flux,
                                                     model,
                                                     (11, 11), 10)
            self._fitTabs.append(tab)
        return self._fitTabs

    def _measurePositionErrorForOneSetOfFrames(self, flux, model):
        tabs = self._fitAllFrames(flux, model)
        dxList = []
        dyList = []
        for tab in tabs:
            dx = self._posX - tab['x_fit']
            dy = self._posY - tab['y_fit']
            dxList.append(dx)
            dyList.append(dy)
        errX = np.array(dxList)
        errY = np.array(dyList)
        return tabs, np.sum(errX**2)/len(tabs), np.sum(errY**2)/len(tabs)

#     def _estimateAstrometricError(self, flux, sx, sy):
#         tabs = self._fitAllFrames(flux, sx, sy)
#         estimator = astrometricError_estimator.EstimateAstrometricError(tabs)
#         estimator.createCubeOfStarsInfo()
#         astromError = estimator.getStandardAstrometricErrorinPixels()
#         return astromError

    def _buildModelAtHighFlux(self):
        highFlux=1e12
        ima = self._createSetOfSameGaussianSource(highFlux,
                                                  howManyImages=1)[0]
        return self._getEPSFOnSingleFrame(ima)

    def measurePositionErrorForDifferentFluxes(self,
                                               fluxVector=None):

        if fluxVector is None:
            fluxVector=np.logspace(3, 8, 30)
        self._model = self._buildModelAtHighFlux()
        resX = []
        resY = []
        self.allTabs = []
        self.fluxVector= fluxVector
        for flux in self.fluxVector:
            print("Computing error for flux %g" % flux)
            tabs, errX, errY = self._measurePositionErrorForOneSetOfFrames(flux,
                                                                           self._model)
            self.allTabs.append(tabs)
            resX.append(errX)
            resY.append(errY)
        self.errXList=np.array(resX)
        self.errYList=np.array(resY)

    def plot(self, color, scale):
        import matplotlib.pyplot as plt
        if scale=='px':
            plt.loglog(
                self.fluxVector, self.errXList, label='FWHM = %g px' % self._fwhm,
                color=color)
            plt.loglog(
                self.fluxVector, self.errYList, '-.', color=color)
            plt.xlabel('PSF Flux [phot]')
            plt.ylabel('Mean Square Error [px]')
            plt.legend()
        elif scale=='mas':
            plt.loglog(
                self.fluxVector, self.errXList*0.119*1e03,
                label='FWHM = %g $^{\prime\prime}$' % (self._fwhm*0.119),
                color=color)
            plt.loglog(
                self.fluxVector, self.errYList*0.119*1e03, '-.', color=color)
            plt.xlabel('PSF Flux [phot]')
            plt.ylabel('Mean Square Error [mas]')
            plt.legend()
