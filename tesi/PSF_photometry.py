'''
Created on 16 feb 2019

@author: gcarla
'''
from photutils.background.core import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.detection.findstars import IRAFStarFinder
from photutils.psf.groupstars import DAOGroup
from photutils.psf.models import IntegratedGaussianPRF
from astropy.modeling.functional_models import Gaussian2D, Moffat2D
from photutils.psf.photometry import BasicPSFPhotometry, DAOPhotPSFPhotometry,\
    IterativelySubtractedPSFPhotometry
from tesi import sandbox


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
        self._iraffinder = sandbox.IRAFStarFinderExcludingMaskedPixel(
            threshold=self._computeThreshold(),
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

    def _setGroupMaker(self):
        self._crit_separation = 2*self._fwhm
        self._groupmaker = DAOGroup(self._crit_separation)
        return self._groupmaker

    def setPSFModel(self, model):
        self._psf_model = model

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
        self._gaussianModel.fluxname= 'amplitude'
        self._gaussianModel.xname= 'x_mean'
        self._gaussianModel.yname= 'y_mean'
        self._psf_model = self._gaussianModel

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

#     def getFitTab(self):
#         return self.basic_result_tab

#     def getFittedPosition(self):
#         return self.basic_result_tab['x_fit', 'y_fit']
#
#     def getFittedFlux(self):
#         return self.basic_result_tab['flux_fit']
#
#     def getResidualImage(self):
#         self.res = self.basic_photom.get_residual_image()
#         return self.res

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
            group_maker=self._setGroupMaker(),
            bkg_estimator=self._bkg_est,
            psf_model=self._psf_model,
            fitter=self._fitter,
            fitshape=self._fitshape,
            aperture_radius=self._apertureRadius,
            niters=niters)
        self.iter_result_tab = self.iterat_photom(self._image)
        return self.iter_result_tab
