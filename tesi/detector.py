import numpy as np
from photutils.utils.check_random_state import check_random_state
from photutils.datasets.make import apply_poisson_noise, \
    make_noise_image


class BaseDetector(object):

    def __init__(self,
                 shape,
                 name,
                 bitDepth,
                 ronInElectrons,
                 quantumEfficiency,
                 darkCurrent,
                 gainAdu2Electrons,
                 biasLevelInAdu,
                 randomState=None):
        """
        Detector specifications

        args:
            name (str): name
            bitDepth (int): bitDepth
            ronInelectrons (float): readout noise in electrons
            quantumEfficiency (float): QE
            darkCurrent (float): dark current in electrons/px/sec
            gainAdu2Electrons (float): gain
            biasLevelInAdu (float): bias
            randomState (mtrand.RandomState): seed (optional)
        """
        self.shape= shape
        self.name= name
        self.gainAdu2Electrons= float(gainAdu2Electrons)
        self.ronInElectrons= float(ronInElectrons)
        self.bitDepth= bitDepth
        self.darkCurrentInElectronsPerPixelPerSecond = \
            float(darkCurrent)
        self.quantumEfficiency= float(quantumEfficiency)
        self.biasLevelInAdu= float(biasLevelInAdu)
        if randomState is None:
            self.seed= randomState
        else:
            self.seed=check_random_state(12345)
        self._exposureTimeInSeconds= 1.0


    def setExposureTime(self, exposureTimeInSeconds):
        """Set the exposure time of the image.

        Args:
            exposureTimeInSeconds (float) : set the exposure time
                in seconds.
        """
        self._exposureTimeInSeconds= exposureTimeInSeconds


    def exposureTimeInSec(self):
        return self._exposureTimeInSeconds


    def biasMapInAdu(self):
        bias_im= np.zeros(self.shape) + self.biasLevelInAdu
        return bias_im


    def readOutNoiseMapInElectrons(self):
        return make_noise_image(
            self.shape,
            type='gaussian',
            mean=0,
            stddev=self.ronInElectrons,
            random_state=self.seed)


    def darkCurrentMapInElectrons(self):
        baseCurrent= self.darkCurrentInElectronsPerPixelPerSecond *\
            self.exposureTimeInSec()
        darkImage= make_noise_image(
            self.shape,
            'poisson',
            baseCurrent,
            0,
            self.seed)
        # TODO: add hot pixels
        return darkImage


    def clipAtSaturation(self, aduImage):
        bitDepth= self.bitDepth
        if bitDepth == np.inf:
            return aduImage
        max_adu= np.int(2**bitDepth - 1)
        aduImageClipped= aduImage.copy()
        aduImageClipped[aduImage > max_adu] = max_adu
        return aduImageClipped


    def _photon2electrons(self, photonImage):
        return np.round(self.quantumEfficiency*photonImage)


    def _electrons2Adu(self, electronImage):
        aduImage= (
            self.biasMapInAdu() +
            (electronImage + self.readOutNoiseMapInElectrons()) /
            self.gainAdu2Electrons).astype(int)
        aduImage= self.clipAtSaturation(aduImage)
        return aduImage


    def photons2Adu(self, photonMap):
        electronImage= self._photon2electrons(photonMap)
        dark= self.darkCurrentMapInElectrons()
        aduImage= self._electrons2Adu(electronImage+dark)
        return aduImage




class IdealDetector(BaseDetector):

    IDEAL="Ideal"

    def __init__(self, shape):
        super().__init__(
            shape,
            name=self.IDEAL,
            bitDepth=np.inf,
            ronInElectrons=0.0,
            quantumEfficiency=1,
            darkCurrent=0,
            gainAdu2Electrons=1,
            biasLevelInAdu=0)


    def photons2Adu(self, photonMap):
        return photonMap


class GenericDetector(BaseDetector):

    def __init__(self, shape):
        super().__init__(
            shape,
            name='Generic CCD',
            bitDepth=12,
            ronInElectrons=20.0,
            quantumEfficiency=0.6,
            darkCurrent=0.1,
            gainAdu2Electrons=1.0,
            biasLevelInAdu=14)



class AvtGC1350Detector(BaseDetector):

    def __init__(self):
        super().__init__(
            (1360, 1024),
            name='AVT GC1350',
            bitDepth=12,
            ronInElectrons=20.0,
            quantumEfficiency=0.6,
            darkCurrent=0.9,
            gainAdu2Electrons=3.9,
            biasLevelInAdu=14)



class LuciDetector(BaseDetector):

    def __init__(self):
        super().__init__(
            (2048, 2048),
            name='Luci',
            bitDepth=16,
            ronInElectrons=10.0,
            quantumEfficiency=0.8,
            darkCurrent=0.006,
            gainAdu2Electrons=2.0,
            biasLevelInAdu=20)


    def biasMapInAdu(self):
        bias_im= np.zeros(self.shape) + self.biasLevelInAdu
        bias_im[::64, :] *= 2.0
        return bias_im



    def _photon2electrons(self, photonImage):
        return np.round(self.quantumEfficiency*photonImage)


    def _electrons2Adu(self, electronImage):
        aduImage= (
            self.biasMapInAdu() +
            (electronImage + super().readOutNoiseMapInElectrons()) /
            self.gainAdu2Electrons).astype(int)
        aduImage= super().clipAtSaturation(aduImage)
        return aduImage


    def photons2Adu(self, photonMap):
        electronImage= super()._photon2electrons(photonMap)
        dark= super().darkCurrentMapInElectrons()
        aduImage= self._electrons2Adu(electronImage+dark)
        return aduImage

