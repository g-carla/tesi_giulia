'''
Created on 18 mar 2019

@author: gcarla
'''
import numpy as np
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from tesi import image_creator, ePSF_builder, image_fitter
from cmath import pi


class testFitPrecision():

    def __init__(self,
                 fwhm,
                 posx=23.5,
                 posy=17.5,
                 shape=(50, 50),
                 ):
        self._fwhm = fwhm
        self._posX = posx
        self._posY = posy
        self._shape = shape
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
        coordStack = np.hstack((
            np.array([np.array(tab['x_fit']) for tab in tabs]),
            np.array([np.array(tab['y_fit']) for tab in tabs])))
        errX = (coordStack[:, 0]).std()
        errY = (coordStack[:, 1]).std()
        return coordStack, np.sqrt(errX**2 + errY**2)

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
        res = []
        self.allCoords = []
        self.fluxVector= fluxVector
        for flux in self.fluxVector:
            print("Computing error for flux %g" % flux)
            coords, err = self._measurePositionErrorForOneSetOfFrames(
                flux, self._model)
            self.allCoords.append(coords)
            res.append(err)
        self.errors=np.array(res)

    def plotErrors(self, color, scale):
        import matplotlib.pyplot as plt
#         plt.loglog(
#             self.fluxVector, self.errors, label='$\sigma_{x}$, $\sigma_{y}$'
#             ' = %g' % self._stdX, color=color)
        #th = np.sqrt(2)*1.491*self._stdX/np.sqrt(self.fluxVector)
        if scale == 'px':
            plt.loglog(self.fluxVector, self.errors,
                       label='FWHM = %g px' % self._fwhm,
                       color=color)
            #plt.loglog(self.fluxVector, th, '-.', label='Theory', color=color)
            plt.xlabel('PSF Flux [phot]')
            plt.ylabel('Standard Deviation [px]')
            plt.legend()
        elif scale == 'mas':
            plt.loglog(self.fluxVector, self.errors*0.119*1e03,
                       label='FWHM = %g $^{\prime\prime}$' % (
                             self._fwhm*0.119),
                       color=color)
            # plt.loglog(self.fluxVector, th*0.119*1e03, '-.', label='Theory',
            #           color=color)
            plt.xlabel('PSF Flux [phot]')
            plt.ylabel('Standard Deviation [mas]')
            plt.legend()
