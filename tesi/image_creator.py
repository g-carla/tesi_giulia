'''
Created on 20 nov 2018

@author: gcarla
'''
import numpy as np
from astropy.table.table import Table
from photutils.datasets.make import make_gaussian_sources_image,\
    apply_poisson_noise, make_model_sources_image
from photutils.utils.check_random_state import check_random_state
from astropy.modeling.functional_models import Gaussian2D
from tesi.detector import IdealDetector
from numpy import float64
from photutils.psf.models import IntegratedGaussianPRF


class ImageCreator(object):

    TRANSMISSION_MAP_IDEAL='TRANSMISSION_IDEAL'
    TRANSMISSION_MAP_REALISTIC='TRANSMISSION_REALISTIC'
    SKY_NO='SKY_NO'
    SKY_IDEAL='SKY_IDEAL'
    SKY_REALISTIC='SKY_REALISTIC'

    def __init__(self,
                 shape,
                 detectorSpecifications=None
                 ):
        self._shape= shape
        self._usePoissonNoise= False
        self._seed=check_random_state(12345)
        self.resetTable()
        if detectorSpecifications is None:
            detectorSpecifications=IdealDetector(self._shape)
        self.setDetector(detectorSpecifications)
        self.setExposureTime(1.0)
        self._skyPhotonsPerPxPerSecond= 100.
        self.setTransmissionMap(self.TRANSMISSION_MAP_IDEAL)
        self.setSkyBackground(self.SKY_NO)

    @property
    def shape(self):
        return self._shape

    def setDetector(self, detectorSpecifications):
        self._detectorSpecifications= detectorSpecifications

    def detector(self):
        """ Return the detector specifications

        Returns:
            detector (DetectorSpecification): the detector
            characteristics used to create the image
        """
        return self._detectorSpecifications

    def setExposureTime(self, exposureTimeInSeconds):
        """Set the exposure time of the image.

        Args:
            exposureTimeInSeconds (float) : set the exposure time
                in seconds.
        """
        self._exposureTimeInSeconds= exposureTimeInSeconds
        self.detector().setExposureTime(exposureTimeInSeconds)

    def exposureTimeInSec(self):
        return self._exposureTimeInSeconds

    def usePoissonNoise(self, trueOrFalse):
        self._usePoissonNoise= trueOrFalse

    def createGaussianImage(self,
                            posX,
                            posY,
                            fluxInPhotons,
                            stdX,
                            stdY=None):
        table = Table()
        table['x_mean']= [posX]
        table['y_mean']= [posY]
        table['x_stddev']= [stdX]
        if stdY is None:
            table['y_stddev']= [stdX]
        else:
            table['y_stddev']= [stdY]
        table['flux']= [fluxInPhotons]
        self._table= table
        ima = np.zeros(self._shape)
        ima += make_gaussian_sources_image(
            self._shape, self._table)
        if self._usePoissonNoise:
            ima= self._addShotNoise(ima)
        return ima
        # return self.createImage()

    def createIntegratedGaussianPRFImage(self,
                                         sigma,
                                         flux,
                                         x0,
                                         y0):
        table = Table()
        table['sigma']= [sigma]
        table['flux']= [flux]
        table['x_0']= [x0]
        table['y_0']= [y0]
        self._table= table
        ima = make_model_sources_image(
            self._shape, IntegratedGaussianPRF(), table)
        return ima

    def createMoffatImage(self,
                          posX,
                          posY,
                          gamma,
                          alpha,
                          peak):
        table = Table()
        table['x_0']= [posX]
        table['y_0']= [posY]
        table['gamma']= [gamma]
        table['alpha']= [alpha]
        table['amplitude']= [peak]
        self._table= table
        return self.createImage()

    def createMultipleIntegratedGaussianPRFImage(
            self,
            stddevRange=[2., 3],
            fluxInPhotons=[1000., 10000],
            nStars=100):
        xMean= np.random.uniform(1, self._shape[1]-1, nStars)
        yMean= np.random.uniform(1, self._shape[0]-1, nStars)
        sx= np.random.uniform(
            stddevRange[0], stddevRange[1], nStars)
        flux= np.random.uniform(
            fluxInPhotons[0],
            fluxInPhotons[1],
            nStars)

        self._table= Table()
        self._table['x_0']= xMean
        self._table['y_0']= yMean
        self._table['sigma']= sx
        self._table['flux']= flux
        ima = make_model_sources_image(
            self._shape, IntegratedGaussianPRF(), self._table)
        return ima

    def createMultipleGaussian(self,
                               stddevXRange=[2., 3],
                               stddevYRange=None,
                               fluxInPhotons=[1000., 10000],
                               nStars=100):
        xMean= np.random.uniform(1, self._shape[1]-1, nStars)
        yMean= np.random.uniform(1, self._shape[0]-1, nStars)
        sx= np.random.uniform(
            stddevXRange[0], stddevXRange[1], nStars)
        if stddevYRange is None:
            sy= sx
        else:
            sy= np.random.uniform(
                stddevYRange[0],
                stddevYRange[1],
                nStars)

        theta= np.arctan2(yMean-0.5*self._shape[0],
                          xMean-0.5*self._shape[1]) - np.pi/2

        flux= np.random.uniform(
            fluxInPhotons[0],
            fluxInPhotons[1],
            nStars)

        self._table= Table()
        self._table['x_mean']= xMean
        self._table['y_mean']= yMean
        self._table['x_stddev']= sx
        self._table['y_stddev']= sy
        self._table['theta']= theta
        self._table['flux']= flux
        ima= self.createImage()
        return ima

    def addGaussianSource(self,
                          posX,
                          posY,
                          stdX,
                          stdY,
                          theta,
                          fluxInPhotons):
        self._table.add_row([posX, posY, stdX, stdY,
                             theta, fluxInPhotons])

    def resetTable(self):
        self._table= Table(
            names=('x_mean', 'y_mean', 'x_stddev', 'y_stddev',
                   'theta', 'flux'))

    def createImage(self):
        """createImage

        Create source images.
        Add transmission and differential sensitivity effects.
        Add shot noise, if asked.
        Convert photons to ADU using the detector model,
        accounting for quantum efficiency, dark current, bias,
        gain, clipping.
        """

        sourceImage= self._createSourcesImage()
        image= self._addSensitivityAndVignetting(sourceImage)
        if self._usePoissonNoise:
            image= self._addShotNoise(image)
        return self.detector().photons2Adu(image)

    def _createSourcesImage(self):
        image= np.zeros(self._shape)
        if self._table is not None:
            image += make_gaussian_sources_image(
                self._shape, self._table)
        return image + self._skyImage*self.exposureTimeInSec()

    def _realisticTransmissionMap(self):
        ima= np.ones(self._shape)
        y, x = np.indices(ima.shape)

        vign_model = Gaussian2D(
            amplitude=1,
            x_mean=self._shape[1] / 2,
            y_mean=self._shape[0] / 2,
            x_stddev=2 * self._shape[1],
            y_stddev=2 * self._shape[0])
        vign_im = vign_model(x, y)
        ima*= vign_im
        return ima

    def _realisticSky(self):

        ima= self._idealSky()
        y, x = np.indices(ima.shape)

        def f(x, y, ampl):
            return y*ampl/y.max() + x* ampl*0.1/x.max()

        ima += f(x, y, ima.mean()*0.1)
        return ima

    def _idealSky(self):
        sky= np.ones(self._shape)* self._skyPhotonsPerPxPerSecond
        return sky

    def setTransmissionMap(self, nameOrMap):
        """setTransmissionMap

        Args:
            nameOrMap (str or ndarray): if nameOrMap is a string
                the corresponding predefined transmission map is used.
                Available names are %s (corresponding to an uniform
                transmission of 1) and and %s (corresponding to
                a vignetting transmission following a gaussian shape).
                If nameOrMap is a numpy array, that array is used.
        Raises:
            KeyError if nameOrMap is a string not corresponding to
            any predefined transmission map.
        """ % (
            self.TRANSMISSION_MAP_IDEAL,
            self.TRANSMISSION_MAP_REALISTIC)

        if isinstance(nameOrMap, str):
            if nameOrMap == self.TRANSMISSION_MAP_IDEAL:
                transmissionMap= np.ones(self._shape)
            elif nameOrMap == \
                    self.TRANSMISSION_MAP_REALISTIC:
                transmissionMap= self._realisticTransmissionMap()
            else:
                raise KeyError('Unknown transmission map %s' %
                               nameOrMap)
            self._transmissionMap= transmissionMap
        else:
            self._transmissionMap= nameOrMap

    def setSkyBackground(self, nameOrMap):
        """setSkyBackground

        Args:
            nameOrMap (str or ndarray): if nameOrMap is a string
                the corresponding predefined sky background is used.
                Available names are %s (no sky background),
                %s (corresponding to a uniform
                background of 100 photons/px/sec) and and %s
                (adding some gradient and spot-like features).
                If nameOrMap is a numpy array, the passed array is used
                as background after scaling by the exposure time
        Raises:
            KeyError if nameOrMap is a string not corresponding to
            any predefined sky background.
        """ % (
            self.SKY_NO,
            self.SKY_IDEAL,
            self.SKY_REALISTIC)

        if isinstance(nameOrMap, str):
            if nameOrMap == self.SKY_NO:
                sky = np.zeros(self._shape)
            elif nameOrMap == self.SKY_IDEAL:
                sky= self._idealSky()
            elif nameOrMap == self.SKY_REALISTIC:
                sky= self._realisticSky()
            else:
                raise KeyError('Unknown sky background map %s' %
                               nameOrMap)
            self._skyImage= sky
        else:
            self._skyImage= nameOrMap

    def _addSensitivityAndVignetting(self, image):
        return image*self._transmissionMap

    def _addShotNoise(self, photonNoNoiseImage):
        return (apply_poisson_noise(
            photonNoNoiseImage, self._seed)).astype(float64)
