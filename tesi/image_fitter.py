'''
Created on 20 nov 2018

@author: gcarla
'''

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from photutils.detection.findstars import IRAFStarFinder
from astropy.modeling import models  # , fitting
from tesi import sandbox
from photutils.background.core import MMMBackground, MADStdBackgroundRMS
from photutils.psf.models import IntegratedGaussianPRF
from astropy.modeling.functional_models import Gaussian2D, Moffat2D
from photutils.psf.photometry import BasicPSFPhotometry, \
    IterativelySubtractedPSFPhotometry
from photutils.psf.groupstars import DAOGroup
from photutils.aperture.circle import CircularAperture


class OneStarFitter(object):
    '''
    '''

    def __init__(self, thresholdInPhot, fwhm, min_separation,
                 sharplo, sharphi, roundlo, roundhi, peakmax=None,
                 fitter=LevMarLSQFitter()):
        self._threshold = thresholdInPhot
        self._fwhm = fwhm
        self._minSep = min_separation
        self._sharplo = sharplo
        self._sharphi = sharphi
        self._roundlo = roundlo
        self._roundhi = roundhi
        self._peakmax = peakmax
        self._fitter = fitter

    def _getInitParameters(self, image):
        self.init_guessTable = None

        dim_y, dim_x = image.shape
        self._y, self._x = np.mgrid[:dim_y, :dim_x]
        iraffind = IRAFStarFinder(threshold=self._threshold, fwhm=self._fwhm,
                                  minsep_fwhm=self._minSep,
                                  sharplo=self._sharplo, sharphi=self._sharphi,
                                  roundlo=self._roundlo, roundhi=self._roundhi,
                                  peakmax=self._peakmax)
        self.init_guessTable = iraffind.find_stars(image)
        if len(self.init_guessTable) == 0:
            raise Exception("No star found - (add info please)")
        elif len(self.init_guessTable) > 1:
            self.init_guessTable = self.init_guessTable[0]

    def gaussianFit(self, image):
        n = gaussian_fwhm_to_sigma
        self._getInitParameters(image)
        fit_model = models.Gaussian2D(x_mean=self.init_guessTable['xcentroid'],
                                      y_mean=self.init_guessTable['ycentroid'],
                                      amplitude=self.init_guessTable['peak'],
                                      x_stddev=self.init_guessTable['fwhm'] * n,
                                      y_stddev=self.init_guessTable['fwhm'] * n
                                      #,theta=init_guessTable['pa']
                                      )
        self._fit = self._fitter(fit_model, self._x, self._y, image)

    def moffatFit(self, image):
        self._getInitParameters(image)
        fit_model = models.Moffat2D(x_0=self.init_guessTable['xcentroid'],
                                    y_0=self.init_guessTable['ycentroid'],
                                    amplitude=self.init_guessTable['peak'])
        self._fit = self._fitter(fit_model, self._x, self._y, image)

    def getFitParameters(self):
        return self._fit

#     def getCentroid(self):
#         para= self.getFitParameters()
#         return np.array([para.x_mean.value[0], para.y_mean.value[0]])
#
#     def getSigmaXY(self):
#         para= self.getFitParameters()
#         return np.array([para.x_stddev.value[0], para.y_stddev.value[0]])
#
#     def getAmplitude(self):
#         para= self.getFitParameters()
#         return para.amplitude.value[0]
#
#     def getFittedImage(self):
#         return self._fit(self._x, self._y)


class StarsFitter(OneStarFitter):

    def __init__(self, image, thresholdInPhot, fwhm, min_separation,
                 sharplo=0.2, sharphi=1., roundlo=-1., roundhi=1.,
                 fitshape=(11, 11), apertureRadius=10, peakmax=5e4,
                 fitter=LevMarLSQFitter(),
                 bkgEstimator=MMMBackground(),
                 bkgRms=MADStdBackgroundRMS()):

        # In the __init__ function you can add more parameters than the ones in
        # the ImageFitter class.
        # In the super__init must be present only the parameters defined in
        # the ImageFitter class.
        # New parameters must be defined below super__init

        super().__init__(thresholdInPhot, fwhm, min_separation,
                         sharplo, sharphi, roundlo, roundhi, peakmax, fitter)
        self._image = image
        self._fitshape = fitshape
        self._apertureRadius = apertureRadius
        self._bkgEst = bkgEstimator
        self._bkgRms = bkgRms

        # Here the definition of ONLY the new parameters

    def fitStars(self, psfModel, niters=0):
        self._setPSFModel(model=psfModel)
        if niters == 0:
            self._doBasicPSFPhotometry()
        elif niters > 0:
            self._doIterativelyPSFPhotometry(niters=niters)

    def getFitTable(self):
        return self._starsTab

    def showFittedStars(self, perc_interval=95, aperture_radius=7):
        from astropy.visualization import PercentileInterval
        positions = (self._starsTab['x_fit'], self._starsTab['y_fit'])
        apertures = CircularAperture(positions, r=aperture_radius)
        sandbox.showNorm(self._image,
                         interval=PercentileInterval(perc_interval))
        apertures.plot()

    def getImageOfResiduals(self):
        return self._basicPhotom.get_residual_image()

    def setImage(self, image):
        self._image = image

    def _setStarsFinder(self):
        self._iraffinder = IRAFStarFinder(
            threshold=self._computeThreshold(),
            fwhm=self._fwhm,
            minsep_fwhm=self._minSep,
            sharplo=self._sharplo,
            sharphi=self._sharphi,
            roundlo=self._roundlo,
            roundhi=self._roundhi,
            peakmax=self._peakmax)
        return self._iraffinder

    def _computeThreshold(self):
        if self._threshold is None:
            return 3.5 * self._bkgrms(self._image)
        else:
            return self._threshold

    def _setPSFModel(self, model):
        if model == 'gaussian':
            self._setGaussianModel()
            self._gaussianModel = Gaussian2D(amplitude=self._amplitude,
                                             x_stddev=self._sx,
                                             y_stddev=self._sy,
                                             x_mean=self._xMean,
                                             y_mean=self._yMean,
                                             theta=self._theta)
            self._gaussianModel.fluxname = 'amplitude'
            self._gaussianModel.xname = 'x_mean'
            self._gaussianModel.yname = 'y_mean'
            self._psfModel = self._gaussianModel
        elif model == 'moffat':
            self._setMoffatModel()
            self._moffatModel = Moffat2D(amplitude=self._amplitude,
                                         x_0=self._x0,
                                         y_0=self._y0,
                                         gamma=self._gamma,
                                         alpha=self._alpha)
            self._moffatModel.fluxname = 'amplitude'
            self._psfModel = self._moffatModel
        elif model == 'integrated gaussian':
            self._setIntGaussianModel()
            self._psfModel = IntegratedGaussianPRF(sigma=self._sigma,
                                                   flux=self._flux,
                                                   x_0=self._xPeak,
                                                   y_0=self._yPeak)
        else:
            self._psfModel = model

    def _setGaussianModel(self, ampl=None, sigmaX=None, sigmaY=None,
                          xMean=None, yMean=None, theta=None):
        self._amplitude = ampl
        self._sx = sigmaX
        self._sy = sigmaY
        self._xMean = xMean
        self._yMean = yMean
        self._theta = theta

    def _setMoffatModel(self, ampl=None, x0=None, y0=None,
                        gamma=None, alpha=None):

        self._amplitude = ampl
        self._x0 = x0
        self._y0 = y0
        self._gamma = gamma
        self._alpha = alpha

    def _setIntGaussianModel(self, flux=None, xPeak=None, yPeak=None):
        self._sigma = self._fwhm * gaussian_fwhm_to_sigma
        self._flux = flux
        self._xPeak = xPeak
        self._yPeak = yPeak

    def _doBasicPSFPhotometry(self):
        self._basicPhotom = BasicPSFPhotometry(
            finder=self._setStarsFinder(),
            group_maker=self._setGroupMaker(),
            bkg_estimator=self._bkgEst,
            psf_model=self._psfModel,
            fitter=self._fitter,
            fitshape=self._fitshape,
            aperture_radius=self._apertureRadius)
        self._starsTab = self._basicPhotom(self._image)

    def _doIterativelyPSFPhotometry(self, niters):
        self._iteratPhotom = IterativelySubtractedPSFPhotometry(
            finder=self._setStarsFinder(),
            group_maker=self._setGroupMaker(),
            bkg_estimator=self._bkgEst,
            psf_model=self._psfModel,
            fitter=self._fitter,
            fitshape=self._fitshape,
            aperture_radius=self._apertureRadius,
            niters=niters)
        self._starsTab = self._iteratPhotom(self._image)

    def _setGroupMaker(self):
        self._critSeparation = 2 * self._fwhm
        self._groupmaker = DAOGroup(self._critSeparation)
        return self._groupmaker
