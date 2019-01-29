'''
Created on 20 nov 2018

@author: gcarla
'''
import numpy as np
from astropy.stats.funcs import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from photutils.detection.findstars import IRAFStarFinder, DAOStarFinder
from astropy.modeling import models, fitting
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.background.core import MMMBackground, MADStdBackgroundRMS
from photutils.psf.groupstars import DAOGroup
from photutils.psf.models import IntegratedGaussianPRF
from photutils.psf import prepare_psf_model
from photutils.psf.photometry import BasicPSFPhotometry, DAOPhotPSFPhotometry,\
    IterativelySubtractedPSFPhotometry
from astropy.modeling.functional_models import Gaussian2D, Moffat2D


class ImageFitter():

    def __init__(self,
                 thresholdInPhotons,
                 fwhm,
                 min_separation,
                 sharplo,
                 sharphi,
                 roundlo,
                 roundhi,
                 peakmax=None
                 # sky
                 ):

        self._threshold = thresholdInPhotons
        self._fwhm = fwhm
        self._min_sep = min_separation
        self._sharplo = sharplo
        self._sharphi = sharphi
        self._roundlo = roundlo
        self._roundhi = roundhi
        self._peakmax = peakmax

    def fitSingleStarWithGaussianFit(self, image):

        dim_y, dim_x = image.shape
        n = gaussian_fwhm_to_sigma
        self._y, self._x = np.mgrid[:dim_y, :dim_x]

        iraffind = IRAFStarFinder(threshold=self._threshold, fwhm=self._fwhm,
                                  minsep_fwhm=self._min_sep,
                                  sharplo=self._sharplo, sharphi=self._sharphi,
                                  roundlo=self._roundlo, roundhi=self._roundhi,
                                  peakmax=self._peakmax)
        self.init_guessTable = iraffind.find_stars(image)

        if len(self.init_guessTable) == 0:
            raise Exception("No star found - (add info please)")
        elif len(self.init_guessTable) > 1:
            self.init_guessTable= self.init_guessTable[0]

        fit_model = models.Gaussian2D(x_mean=self.init_guessTable['xcentroid'],
                                      y_mean=self.init_guessTable['ycentroid'],
                                      amplitude=self.init_guessTable['peak'],
                                      x_stddev=self.init_guessTable['fwhm']*n,
                                      y_stddev=self.init_guessTable['fwhm']*n
                                      #,theta=init_guessTable['pa']
                                      )
        fitter = fitting.LevMarLSQFitter()
        self._fit = fitter(fit_model, self._x, self._y, image)

    def fitStarsWithGaussianFit(self, image):

        dim_y, dim_x = image.shape
        n = gaussian_fwhm_to_sigma
        self._y, self._x = np.mgrid[:dim_y, :dim_x]

        iraffind = IRAFStarFinder(threshold=self._threshold, fwhm=self._fwhm,
                                  minsep_fwhm=self._min_sep,
                                  sharplo=self._sharplo, sharphi=self._sharphi,
                                  roundlo=self._roundlo, roundhi=self._roundhi)
        self.init_guessTable = iraffind.find_stars(image)

        if len(self.init_guessTable) == 0:
            raise Exception("No star found - (add info please)")

    def fitSingleStarWithCentroid(self, ima):
        sy, sx= ima.shape
        y, x= np.mgrid[0:sy, 0:sx]
        cx=np.sum(ima*x)/np.sum(ima)
        cy=np.sum(ima*y)/np.sum(ima)
        self._fit= models.Gaussian2D(x_mean=(cx,), y_mean=(cy,),
                                     amplitude=(np.nan,), x_stddev=(np.nan,),
                                     y_stddev=(np.nan,))

    def getFitParameters(self):
        return self._fit

    def getCentroid(self):
        para= self.getFitParameters()
        return np.array([para.x_mean.value[0], para.y_mean.value[0]])

    def getSigmaXY(self):
        para= self.getFitParameters()
        return np.array([para.x_stddev.value[0], para.y_stddev.value[0]])

    def getAmplitude(self):
        para= self.getFitParameters()
        return para.amplitude.value[0]

    def getFittedImage(self):
        return self._fit(self._x, self._y)


class PSFphotometry():

    def __init__(self,
                 fitshape,
                 aperture_radius):

        self._fitshape=fitshape
        self._apertureRadius = aperture_radius
        self._image = None
        self._bkg_est = MMMBackground()
        self._bkgrms = MADStdBackgroundRMS()
        self._fitter = LevMarLSQFitter()
#         self._moffatModel= None
#         self._gaussianModel = None
#         self._psf_model = None
        # self._initGroupMaker()
        # self._initPSFmodel()

    def setFinder(self,
                  thresholdInPhotons,
                  fwhm,
                  min_separation,
                  sharplo,
                  sharphi,
                  roundlo,
                  roundhi,
                  peakMax):

        self._threshold = thresholdInPhotons
        self._fwhm = fwhm
        self._min_separation = min_separation
        self._sharplo=sharplo
        self._sharphi=sharphi
        self._roundlo=roundlo
        self._roundhi=roundhi
        self._peakmax = peakMax
        self._iraffinder = IRAFStarFinder(threshold=self._computeThreshold(),
                                          fwhm=self._fwhm,
                                          minsep_fwhm=self._min_separation,
                                          sharplo=self._sharplo,
                                          sharphi=self._sharphi,
                                          roundlo=self._roundlo,
                                          roundhi=self._roundhi,
                                          peakmax=self._peakmax)

    def _computeThreshold(self):
        if self._threshold is None:
            return 3.5*self._bkgrms(self._image)
        else:
            return self._threshold

    def setImage(self, image):
        self._image = image
        # self._iraffinder

#     def _initFinder(self):
#         self._iraffinder = IRAFStarFinder(threshold=self._computeThreshold(),
#                                           fwhm=self._fwhm,
#                                           minsep_fwhm=0.01,
#                                           sharplo=self._sharplo,
#                                           sharphi=self._sharphi,
#                                           roundlo=self._roundlo,
#                                           roundhi=self._roundhi,
#                                           peakmax=self._peakmax)
#         return self._iraffinder

    def _setGroupMaker(self):
        self._crit_separation = 2*self._fwhm
        self._groupmaker = DAOGroup(self._crit_separation)
        return self._groupmaker

    def setIntegratedGaussianPRFModel(self,
                                      sigma,
                                      flux=None,
                                      xPeak=None,
                                      yPeak=None):
        #self._sigma = self._fwhm * gaussian_fwhm_to_sigma
        self._psf_model = IntegratedGaussianPRF(sigma=sigma,
                                                flux=flux,
                                                x_0=xPeak,
                                                y_0=yPeak)

    def setGaussian2DModel(self,
                           ampl=None,
                           sigmaX=None,
                           sigmaY=None,
                           xMean=None,
                           yMean=None,
                           theta=None):

        self._gaussianModel = Gaussian2D(amplitude=ampl,
                                         x_stddev=sigmaX,
                                         y_stddev=sigmaY,
                                         x_mean=xMean,
                                         y_mean=yMean,
                                         theta=theta)
        self._psf_model = prepare_psf_model(psfmodel=self._gaussianModel)

    def setMoffat2DModel(self,
                         ampl=None,
                         xPeak=None,
                         yPeak=None,
                         gamma=None,
                         alpha=None):

        self._amplitude = ampl
        self._xPeak = xPeak
        self._yPeak = yPeak
        self._gamma = gamma
        self._alpha = alpha
        self._moffatModel = Moffat2D(amplitude=self._amplitude,
                                     x_0=self._xPeak,
                                     y_0=self._yPeak,
                                     gamma=self._gamma,
                                     alpha=self._alpha)
        self._moffatModel.fluxname= 'amplitude'
        self._psf_model = self._moffatModel

#     def _initPSFmodel(self):
#         self._psf_model = IntegratedGaussianPRF(sigma=self._sigma_psf)
#                  gaussian2D_model = Gaussian2D(x_stddev=2.0, y_stddev=2.0)
#                  self._psf_model = prepare_psf_model(psfmodel=gaussian2D_model)
#          xname='x_mean', yname='y_mean',
#          fluxname='amplitude')
#         moffat2D_model = Moffat2D(alpha=3.)
#         self._psf_model = prepare_psf_model(psfmodel=moffat2D_model)
#         return self._psf_model

    def basicPSFphotometry(self):
        self.basic_photom = BasicPSFPhotometry(
            finder=self._iraffinder,
            group_maker=self._setGroupMaker(),
            bkg_estimator=self._bkg_est,
            psf_model=self._psf_model,
            fitter=self._fitter,
            fitshape=self._fitshape,
            aperture_radius=self._apertureRadius)
        self.basic_result_tab = self.basic_photom(self._image)
        return self.basic_result_tab

    def daoPSFphotometry(self, niters):
        self.dao_photom = DAOPhotPSFPhotometry(
            crit_separation=self._crit_separation,
            threshold=self._computeThreshold(),
            fwhm=self._fwhm,
            psf_model=self._psf_model,
            fitshape=self._fitshape,
            sharplo=self._sharplo,
            sharphi=self._sharphi,
            roundlo=self._roundlo,
            roundhi=self._roundhi,
            fitter=self._fitter,
            aperture_radius=self._apertureRadius,
            niters=niters)
        self.dao_result_tab = self.dao_photom(self._image)
        return self.dao_result_tab

    def iterativelyPSFphotometry(self, niters):
        self.iterat_photom = IterativelySubtractedPSFPhotometry(
            finder=self._iraffinder,
            group_maker=self._groupmaker,
            bkg_estimator=self._bkg_est,
            psf_model=self._psf_model,
            fitter=self._fitter,
            fitshape=self._fitshape,
            aperture_radius=self._apertureRadius,
            niters=niters)
        self.iter_result_tab = self.iterat_photom(self._image)
        return self.iter_result_tab
