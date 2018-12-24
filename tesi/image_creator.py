'''
Created on 20 nov 2018

@author: gcarla
'''
import numpy as np
from astropy.table.table import Table
from photutils.datasets.make import make_gaussian_sources_image,\
    apply_poisson_noise, make_noise_image
from photutils.utils.check_random_state import check_random_state


class DetectorSpecifications(object):

    def __init__(self,
                 name='a CCD',
                 bitDepth=16,
                 ronInElectrons=0.0,
                 quantumEfficiency=0.7,
                 darkCurrent=10.0,
                 gainAdu2Electrons=2,
                 biasLevelInAdu=20.):
        """
        Detector specifications
        """
        self.name= name
        self.gainAdu2Electrons= float(gainAdu2Electrons)
        self.ronInElectrons= float(ronInElectrons)
        self.bitDepth= bitDepth
        self.darkCurrentInElectronsPerPixelPerSecond = \
            float(darkCurrent)
        self.quantumEfficiency= float(quantumEfficiency)
        self.biasLevelInAdu= float(biasLevelInAdu)


    @staticmethod
    def ideal():
        return DetectorSpecifications(
            name='ideal',
            bitDepth=np.inf,
            ronInElectrons=0.0,
            quantumEfficiency=1,
            darkCurrent=0,
            gainAdu2Electrons=1,
            biasLevelInAdu=0)


    @staticmethod
    def avtGc1350():
        return DetectorSpecifications(
            name='AVT GC1350',
            bitDepth=12,
            ronInElectrons=20.0,
            quantumEfficiency=1,
            darkCurrent=0.9,
            gainAdu2Electrons=3.9,
            biasLevelInAdu=14)



class ImageCreator(object):

    def __init__(self,
                 shape,
                 detectorSpecifications=None
                 ):
        self._shape= shape
        self._usePoissonNoise= False
        self._seed=check_random_state(12345)
        self._table= None
        if detectorSpecifications is None:
            detectorSpecifications=DetectorSpecifications.ideal()
        self.setDetector(detectorSpecifications)
        self._exposureTimeInSeconds= 1.0

    def setDetector(self, detectorSpecifications):
        self._detectorSpecifications= detectorSpecifications


    def detector(self):
        return self._detectorSpecifications


    def setExposureTime(self, exposureTimeInSeconds):
        """
        Set the exposure time of the image.

        Args:
            exposureTimeInSeconds (float) : set the exposure time
                in seconds.
        """
        self._exposureTimeInSeconds= exposureTimeInSeconds


    def exposureTimeInSec(self):
        return self._exposureTimeInSeconds


    def setReadOutNoise(self, ronInElectrons):
        self._ronInElectrons= float(ronInElectrons)


    def getReadOutNoise(self):
        return self._ronInElectrons


    def setDarkCurrent(self, darkCurrentInElectronsPerPixelPerSecond):
        self._darkCurrentInElectronsPerPixelPerSecond= \
            float(darkCurrentInElectronsPerPixelPerSecond)


    def getDarkCurrent(self):
        return self._darkCurrentInElectronsPerPixelPerSecond


    def setBiasLevel(self, biasLevelInAdu):
        self._biasLevelInAdu= float(biasLevelInAdu)


    def getBiasLevel(self):
        return self._biasLevelInAdu


    def usePoissonNoise(self, trueOrFalse):
        self._usePoissonNoise= trueOrFalse



    def mimickIdealDetector(self):
        '''
        Mimick the behaviour of an ideal detector
        No dark current, no ron, gain=1, bias=0, QE=1
        And more than that: no Poisson noise too!
        '''
        self.setDarkCurrent(0)
        self.setReadOutNoise(0)
        self.usePoissonNoise(False)
        self.setBiasLevel(0)
        self.setQuantumEfficiec

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

        amp= np.random.uniform(
            fluxInPhotons[0],
            fluxInPhotons[1],
            nStars)

        self._table = Table()
        self._table['x_mean']= xMean
        self._table['y_mean']= yMean
        self._table['x_stddev']= sx
        self._table['y_stddev']= sy
        self._table['amplitude']= amp
        ima= make_gaussian_sources_image(self._shape, self._table)
        if self._usePoissonNoise:
            ima= apply_poisson_noise(ima, random_state=self._seed)
        if self._ronInElectrons != 0:
            ron= self._readOutNoise()
            ima= ima + ron
        return ima


    def getTable(self):
        return self._table


    def createGaussianImageOld(self,
                            posX,
                            posY,
                            stdX,
                            stdY,
                            fluxInPhotons):
        table = Table()
        table['x_mean']= [posX]
        table['y_mean']= [posY]
        table['x_stddev']= [stdX]
        table['y_stddev']= [stdY]
        table['flux']= [fluxInPhotons]
        ima= make_gaussian_sources_image(self._shape, table)
        if self._usePoissonNoise:
            ima= apply_poisson_noise(ima, random_state=self._seed)
        if self._ronInElectrons != 0:
            ron= self._readOutNoise()
            ima= ima + ron
        return ima


    def createGaussianImage(self,
                            posX,
                            posY,
                            stdX,
                            stdY,
                            fluxInPhotons):
        table = Table()
        table['x_mean']= [posX]
        table['y_mean']= [posY]
        table['x_stddev']= [stdX]
        table['y_stddev']= [stdY]
        table['flux']= [fluxInPhotons]
        self._table= table
        return self.createImage()


    def createImage(self):
        """createImage

        Create source images.
        Add transmission and differential sensitivity effects.
        Add shot noise, if asked.
        If ccd name is 'ideal', return the photon map.
        Otherwise convert photons to ADU using the detector model,
        accounting for quantum efficiency, dark current, bias,
        gain, clipping.
        """

        sourceImage= self._createSourcesImage()
        image= self._addSensitivityAndVignetting(sourceImage)
        if self._usePoissonNoise:
            image= self._addShotNoise(image)
        if self.detector().name == 'ideal':
            return image
        electronImage= self._photon2electrons(image)
        dark= self._darkCurrent()
        aduImage= self._electrons2Adu(electronImage+dark)
        return aduImage


    def _createSourcesImage(self):
        image= np.zeros(self._shape)
        if self._table is not None:
            image += make_gaussian_sources_image(
                self._shape, self._table)
        skyImage= 0
        return image + skyImage


    def _addSensitivityAndVignetting(self, image):
        # create transparency Map
        transmissionMap= np.ones(self._shape)
        return image*transmissionMap


    def _addShotNoise(self, photonNoNoiseImage):
        return apply_poisson_noise(
            photonNoNoiseImage, self._seed)


    def _photon2electrons(self, photonImage):
        return np.round(self.detector().quantumEfficiency*photonImage)


    def _electrons2Adu(self, electronImage):
        aduImage= (
            self._bias() +
            (electronImage + self._readOutNoiseInElectrons()) /
            self.detector().gainAdu2Electrons).astype(int)
        aduImage= self._clipAtSaturation(aduImage)
        return aduImage


    def _clipAtSaturation(self, aduImage):
        bitDepth= self.detector().bitDepth
        if bitDepth == np.inf:
            return aduImage
        max_adu= np.int(2**bitDepth - 1)
        aduImageClipped= aduImage.copy()
        aduImageClipped[aduImage > max_adu] = max_adu
        return aduImageClipped


    def _readOutNoiseInElectrons(self):
        return make_noise_image(
            self._shape,
            type='gaussian',
            mean=0,
            stddev=self.detector().ronInElectrons,
            random_state=self._seed)


    def _bias(self):
        bias_im= np.zeros(self._shape) + self.detector().biasLevelInAdu
        # add stripes, colums etc
        return bias_im


    def _darkCurrent(self):
        dark= self.detector().darkCurrentInElectronsPerPixelPerSecond
        baseCurrent = dark * self.exposureTimeInSec()
        darkImage= make_noise_image(
            self._shape, 'poisson', baseCurrent, 0, self._seed)
        # add hot pixels
        return darkImage

