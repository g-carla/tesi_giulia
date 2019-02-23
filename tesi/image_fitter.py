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
from tesi import PSF_photometry, sandbox
from photutils.aperture.circle import CircularAperture


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
        self._fitter = LevMarLSQFitter()

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
        self._fit = self._fitter(fit_model, self._x, self._y, image)

    def fitSingleStarWithMoffatFit(self, image):

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

        fit_model = models.Moffat2D(x_0=self.init_guessTable['xcentroid'],
                                    y_0=self.init_guessTable['ycentroid'],
                                    amplitude=self.init_guessTable['peak'])
        self._fit = self._fitter(fit_model, self._x, self._y, image)


#     def fitStarsWithGaussianFit(self, image):
#
#         dim_y, dim_x = image.shape
#         n = gaussian_fwhm_to_sigma
#         self._y, self._x = np.mgrid[:dim_y, :dim_x]
#
#         iraffind = IRAFStarFinder(threshold=self._threshold, fwhm=self._fwhm,
#                                   minsep_fwhm=self._min_sep,
#                                   sharplo=self._sharplo, sharphi=self._sharphi,
#                                   roundlo=self._roundlo, roundhi=self._roundhi)
#         self.init_guessTable = iraffind.find_stars(image)
#
#         if len(self.init_guessTable) == 0:
#             raise Exception("No star found - (add info please)")

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

    def fitStarsWithBasicPhotometry(self,
                                    image,
                                    model,
                                    fitshape,
                                    apertureRadius):
        self._psfFitter = PSF_photometry.PSFphotometry(
            fitshape, apertureRadius)
        self._psfFitter.setFinder(self._threshold, self._fwhm, self._min_sep,
                                  self._sharplo, self._sharphi,
                                  self._roundlo, self._roundhi,
                                  self._peakmax)
        self._psfFitter.setImage(image)
        self._psfFitter.setPSFModel(model)
        self.fitTab = self._psfFitter.basicPSFphotometry()

    def fitStarsWithIteratedPhotometry(self,
                                       image,
                                       model,
                                       fitshape,
                                       apertureRadius,
                                       niters):
        self._psfFitter = PSF_photometry.PSFphotometry(
            fitshape, apertureRadius)
        self._psfFitter.setFinder(self._threshold, self._fwhm, self._min_sep,
                                  self._sharplo, self._sharphi,
                                  self._roundlo, self._roundhi,
                                  self._peakmax)
        self._psfFitter.setImage(image)
        self._psfFitter.setPSFModel(model)
        self.fitTab = self._psfFitter.iterativelyPSFphotometry(niters)

    def getFitTable(self):
        return self.fitTab

    def showFoundStars(self, image, aperture_radius=7):
        positions = (self.fitTab['x_fit'], self.fitTab['y_fit'])
        apertures = CircularAperture(positions, r=aperture_radius)
        sandbox.showNorm(image)
        apertures.plot()

    def getImageOfResiduals(self):
        return self._psfFitter.basic_photom.get_residual_image()
