'''
Created on 12 feb 2019

@author: gcarla
'''
from photutils.detection.findstars import IRAFStarFinder
from photutils.aperture.circle import CircularAperture
from tesi import sandbox
import matplotlib.pyplot as plt


class StarFinder():
    '''
    '''

    def __init__(self,
                 threshold,
                 fwhm,
                 minSep=1.,
                 sharplo=0.1,
                 sharphi=2.0,
                 roundlo=-1.0,
                 roundhi=1.0,
                 peakmax=5e04):
        self.threshold = threshold
        self.fwhm = fwhm
        self.minSep = minSep
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.peakmax = peakmax

    def _setThreshold(self, threshold):
        self.threshold = threshold

    def _setFWHM(self, fwhm):
        self.fwhm = fwhm

    def _setMinimumSeparation(self, minSep):
        self.minSep = minSep

    def _setSharpnessLowLimit(self, sharplo):
        self.sharplo = sharplo

    def _setSharpnessHighLimit(self, sharphi):
        self.sharphi = sharphi

    def _setRoundnessLowLimit(self, roundlo):
        self.roundlo = roundlo

    def _setRoundnessHighLimit(self, roundhi):
        self.roundhi = roundhi

    def _setMaximumPeak(self, peakmax):
        self.peakmax = peakmax

    def _setFinder(self):
        self.finder = IRAFStarFinder(threshold=self.threshold,
                                     fwhm=self.fwhm,
                                     minsep_fwhm=self.minSep,
                                     sharplo=self.sharplo,
                                     sharphi=self.sharphi,
                                     roundlo=self.roundlo,
                                     roundhi=self.roundhi,
                                     peakmax=self.peakmax)
        return self.finder

    def getFoundStarsTable(self, ima, mask=None):
        self.starsTable = self._setFinder().find_stars(ima, mask)
        return self.starsTable

    def showFoundStars(self, ima, color, aperture_radius=7):
        tab = self.getFoundStarsTable(ima)
        positions = (tab['xcentroid'], tab['ycentroid'])
        apertures = CircularAperture(positions, r=aperture_radius)
        sandbox.showNorm(ima)
        apertures.plot(color=color)

    def showFoundStarsInLUCIFoV(self, ima, color, aperture_radius=7):
        tab = self.getFoundStarsTable(ima)
        positions = (tab['xcentroid'], tab['ycentroid'])
        apertures = CircularAperture(positions, r=aperture_radius)
        sandbox.showNorm(ima, extent=[-120, 120, -120, 120])
        apertures.plot(color=color)
        plt.xlabel('arcsec', size=12)
        plt.ylabel('arcsec', size=12)
