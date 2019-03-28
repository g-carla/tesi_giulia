'''
Created on 12 feb 2019

@author: gcarla
'''


from astropy.table.table import Table
from photutils.psf.epsf_stars import extract_stars, EPSFStars, EPSFStar
from astropy.nddata.nddata import NDData
from photutils.psf.epsf import EPSFBuilder
from tesi import star_finder, sandbox
import matplotlib.pyplot as plt
from tesi import sandbox
from photutils.detection.findstars import IRAFStarFinder
from photutils.background.core import MMMBackground


class ePSFBuilder():
    '''

    '''

    def __init__(self,
                 image,
                 threshold,
                 fwhm,
                 minSep=10.,
                 sharplo=0.1,
                 sharphi=2.0,
                 roundlo=-1.0,
                 roundhi=1.0,
                 peakmax=5e04,
                 size=50):
        self.image = image
        self.threshold = threshold
        self.fwhm = fwhm
        self.minSep = minSep
        self.sharplo = sharplo
        self.sharphi = sharphi
        self.roundlo = roundlo
        self.roundhi = roundhi
        self.peakmax = peakmax
        self.size = size

    def _setImage(self, image):
        self.image = image

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

    def _findStars(self):
        self._finder = sandbox.IRAFStarFinderExcludingMaskedPixel(
            threshold=self.threshold,
            fwhm=self.fwhm,
            minsep_fwhm=self.minSep,
            sharplo=self.sharplo,
            sharphi=self.sharphi,
            roundlo=self.roundlo,
            roundhi=self.roundhi,
            peakmax=self.peakmax)
        self.selectedStars = self._finder.find_stars(self.image)
        return self.selectedStars

    def _setSizeOfExtractionRegion(self, size):
        self.size = size

    def removeBackground(self):
        bkg_est = MMMBackground()
        self.image -= bkg_est(self.image)

    def extractStars(self):
        self.extrTable = Table()
        self.extrTable['x'] = self._findStars()['xcentroid']
        self.extrTable['y'] = self._findStars()['ycentroid']
        # TODO: BACKGROUND SUBTRACTION?
        #meanVal, medVal, stdVal = sigma_clipped_stats(self.image, sigma=3.)
        self.ePSFstars = extract_stars(NDData(self.image),
                                       self.extrTable,
                                       self.size)

    def selectGoodStars(self):
        self.goodStars = []
        for i in range(len(self.ePSFstars)):
            sandbox.showNorm(self.ePSFstars[i].data)
            print('Do you want to keep the star?')
            answer=input()
            if answer=='y':
                self.goodStars.append(self.ePSFstars[i])
            else:
                print('Star is not good')
            plt.close()
        self.ePSFstars = EPSFStars(self.goodStars)
        print('N good stars: %d' %len(self.ePSFstars))

    def buildEPSF(self, oversampling=1, maxiters=10, **kwargs):
        # TODO: add ePSFBuilder parameter (shape, smooting_kernel,
        # recentering_func...)
        self.epsf_builder = EPSFBuilder(oversampling=oversampling,
                                        maxiters=maxiters,
                                        progress_bar=False, **kwargs)
        self.epsfModel, self.fittedStars = self.epsf_builder(
            self.ePSFstars)

    def getEPSFModel(self):
        '''
        EPSFModel object
        Returns the construced ePSF with 
        param_names = ('flux', 'x_0', 'y_0')
        '''
        return self.epsfModel

    def getEPSFImage(self):
        '''
        numpy.ndarray object
        Returns the image of the constructed ePSF
        '''
        return self.epsfModel.data

    def getFittedStars(self):
        '''
        EPSFStars object
        Returns the input stars with updated centers and fluxes derived
        from fitting the output ``epsf``
        '''
        return self.fittedStars

    def showSelectedStarsOnIma(self):
        return self._finder.showFoundStars(self.image, 'r', 10)
