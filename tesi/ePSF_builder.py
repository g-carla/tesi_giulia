'''
Created on 12 feb 2019

@author: gcarla
'''

from tesi import sandbox
from astropy.table.table import Table
from photutils.psf import extract_stars
from photutils.psf.epsf_stars import EPSFStars
from astropy.nddata.nddata import NDData
from photutils.psf.epsf import EPSFBuilder
import matplotlib.pyplot as plt
from photutils.background.core import MMMBackground


class epsfBuilder():
    '''
    Build the "effective Point Spread Function" (see Anderson & King, 2000)
    of the image using a set of sources whose selection depends on the
    IRAFStarFinder parameters choosen by the user.
    The ePSF is built using photutils class "EPSFBuilder"
    '''

    def __init__(self,
                 image,
                 threshold,
                 fwhmInPix,
                 minSep=10.,
                 sharplo=0.1,
                 sharphi=2.0,
                 roundlo=-1.0,
                 roundhi=1.0,
                 peakmax=5e04,
                 size=50):
        self._image = image
        self._threshold = threshold
        self._fwhm = fwhmInPix
        self._minSep = minSep
        self._sharplo = sharplo
        self._sharphi = sharphi
        self._roundlo = roundlo
        self._roundhi = roundhi
        self._peakmax = peakmax
        self._size = size
        self._ePSFstars = None

    def setImage(self, image):
        self._image = image

    def removeBackground(self):
        bkg_est = MMMBackground()
        self._image -= bkg_est(self._image)

    def buildEPSF(self, oversampling=1, maxiters=10, recenteringMaxIters=20,
                  **kwargs):
        # TODO: add ePSFBuilder parameter (shape, smoothing_kernel...)
        if self._ePSFstars is None:
            self._selectGoodStars()
        self._epsfBuilder = EPSFBuilder(
            oversampling=oversampling,
            maxiters=maxiters,
            progress_bar=False,
            recentering_maxiters=recenteringMaxIters,
            **kwargs)
        self._epsfModel, self._fittedStars = self._epsfBuilder(
            self._ePSFstars)

    def getEPSFModel(self):
        '''
        Return the built ePSF as EPSFModel with
        param_names = ('flux', 'x_0', 'y_0').
        '''
        return self._epsfModel

    def getEPSFImage(self):
        '''
        Return the image of the ePSF as a numpy.ndarray object.
        '''
        return self._epsfModel.data

    def getFittedStars(self):
        '''
        Return the input stars with updated position and flux values
        as an EPSFStars object.
        Positions and fluxes are obtained from a least squares
        fit using the new EPSF as model .
        '''
        return self._fittedStars

    def _findStars(self):
        self._finder = sandbox.IRAFStarFinderExcludingMaskedPixel(
            threshold=self._threshold,
            fwhm=self._fwhm,
            minsep_fwhm=self._minSep,
            sharplo=self._sharplo,
            sharphi=self._sharphi,
            roundlo=self._roundlo,
            roundhi=self._roundhi,
            peakmax=self._peakmax)
        self._selectedStars = self._finder.find_stars(self._image)

    def _extractStars(self):
        self._findStars()
        self._starsTab = Table()
        self._starsTab['x'] = self._selectedStars['xcentroid']
        self._starsTab['y'] = self._selectedStars['ycentroid']
        self._starsCut = extract_stars(NDData(data=self._image),
                                       self._starsTab,
                                       self._size)

    def _selectGoodStars(self):
        self._extractStars()
        self._goodStars = []
        for i in range(len(self._starsCut)):
            sandbox.showNorm(self._starsCut[i].data)
            print('Do you want to keep the star? (y/n)')
            answer=input()
            if answer=='y':
                self._goodStars.append(self._starsCut[i])
            else:
                print('Star is not good')
            plt.close()
        self._ePSFstars = EPSFStars(self._goodStars)
        print('N good stars: %d' %len(self._ePSFstars))
