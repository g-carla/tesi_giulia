'''
Created on 16 ott 2018

@author: gcarla
'''

import numpy as np
import ccdproc
from astropy import units as u
from ccdproc.combiner import Combiner
from astropy.modeling import models


class ImageCleaner():
    '''
    Reduce astronomical images performing dark subtraction, flatfield
    correction and sky subtraction.

    Parameters
    ----------
    darkImas, flatImas, skyImas: list of dark, flat and sky images as CCDData.
    sciImas: single scientific image as CCDData or list of scientific images as
        CCDData.
    '''

    def __init__(self, darkImas, flatImas, skyImas, sciImas):
        self._darks = darkImas
        self._flats = flatImas
        self._skies = skyImas
        self._science = sciImas
        self.masterDark = None
        self.masterFlat = None
        self.masterSky = None
        self.masterSci = None

    def setScienceImage(self, ima):
        self._science = ima

    def setMasterDark(self, mDark):
        self.masterDark = mDark

    def setMasterFlat(self, mFlat):
        self.masterFlat = mFlat

    def setMasterSky(self, mSky):
        self.masterSky = mSky

    def getScienceImage(self, unit='electron'):
        '''
        Return the scientific image after the calibration.

        Parameters
        ----------
        unit: 'electron' or 'ADU'. Default is 'electron'.

        Return
        ------
        masterSci: reduced scientfic image.
        '''
        self._unit = unit
        self._computeScienceImage()
        return self.masterSci

    def getSkyImage(self):
        if self.masterSky is None:
            self._computeSkyImage()
        return self.masterSky

    def getFlatImage(self):
        if self.masterFlat is None:
            self._computeFlatImage()
        return self.masterFlat

    def getDarkImage(self):
        if self.masterDark is None:
            self._computeDarkImage()
        return self.masterDark

    def _computeScienceImage(self):
        #        self.sciTrim = self._overscanAndtrim(self.science)
        mSky = self.getSkyImage()
        if type(self._science) == list:
            sciCorrected = []
            for sci in self._science:
                sciDark = self._subtractDark(sci)
                sciFlat = self._correctForFlat(sciDark)
                sciSky = sciFlat.subtract(mSky)
                sciCorrected.append(sciSky)
            self._sciCombiner = Combiner(sciCorrected)
            self.masterSci = self._sciCombiner.median_combine()
            print('Writing the header')
            self.masterSci.header['FRAMETYP'] = \
                self._science[0].header['FRAMETYP']
            self.masterSci.header['OBJECT'] = self._science[0].header['OBJECT']
            self.masterSci.header['DIT'] = self._science[0].header['DIT']
            self.masterSci.header['FILTER'] = \
                self._science[0].header['FILTER']
            self.masterSci.header['OBJRA'] = self._science[0].header['OBJRA']
            self.masterSci.header['OBJDEC'] = self._science[0].header['OBJDEC']
            self.masterSci.header['DATE'] = self._science[0].header['DATE']
            self.masterSci.header['GAIN'] = self._science[0].header['GAIN']
        else:
            sci_dark = self._subtractDark(self._science)
            sciFlat = self._correctForFlat(sci_dark)
            self.masterSci = sciFlat.subtract(mSky)
            # self.masterSci = cosmicray_lacosmic(self.sci_sky,
            #                                        sigclip=5)
            print('Writing the header')
            self.masterSci.header = self._science.header

        if self._unit == 'electron':
            self.masterSci = self._adu2Electron(self.masterSci)
            self.masterSci.header['UNIT'] = 'electrons'

    def _computeSkyImage(self):
        print('Getting masterSky')
#        self.skyTrim = self._overscanAndtrim(self.skies)
        self._skiesCorrected = []
        for sky in self._skies:
            skyDark = self._subtractDark(sky)
            skyFlat = self._correctForFlat(skyDark)
            self._skiesCorrected.append(skyFlat)
        self._skyCombiner = Combiner(self._skiesCorrected)
        self._iterativeSigmaClipping(self._skyCombiner, 1)
        self.masterSky = self._skyCombiner.median_combine()

    def _computeFlatImage(self):
        print('Getting masterFlat')
#       self.flatsTrim = self._overscanAndtrim(self.flats)
        self._flatsDarkSubtracted = []
        for flat in self._flats:
            flat = self._subtractDark(flat)
            self._flatsDarkSubtracted.append(flat)
        self._flatCombiner= self._makeClippedCombiner(
            self._flatsDarkSubtracted)
        scalingFunc = lambda arr: 1/np.ma.average(arr)
        self._flatCombiner.scaling = scalingFunc
        self.masterFlat = self._flatCombiner.median_combine()
        self.masterFlat.header = self._flatsDarkSubtracted[0].meta
#        self.masterFlatInElectrons = self._adu2Electron(self.masterFlat)
        if self.masterFlat.data.min() == 0:
            i = np.argwhere(self.masterFlat.data==0)
            y = i[:, 0]
            x = i[:, 1]
            self.masterFlat.data[y, x] = 0.1
        elif self.masterFlat.data.min() < 0:
            i = np.argwhere(self.masterFlat.data==0)
            y = i[:, 0]
            x = i[:, 1]
            self.masterFlat.data[y, x] = 0.1

    def _computeDarkImage(self):
        print('Getting masterDark')
#        self.darksTrim = self._overscanAndtrim(self.darks)
        self._darkCombiner = self._makeClippedCombiner(self._darks)
        self.masterDark = self._darkCombiner.median_combine()
        self.masterDark.header[
            'DIT'] = self._darkCombiner.ccd_list[0].header['DIT']
        self.masterDark.header['RDNOISE'] = self._darkCombiner.ccd_list[
            0].header['RDNOISE']
        self.masterDark.header[
            'GAIN'] = self._darkCombiner.ccd_list[0].header['GAIN']
        self.masterDark.header['DATASEC'] = self._darkCombiner.ccd_list[
            0].header['DATASEC']

    def _subtractDark(self, ccd):
        mDark = self.getDarkImage()
        print('Subtracting masterDark')
        ccd = ccdproc.subtract_dark(
            ccd,
            mDark,
            exposure_time='DIT',
            exposure_unit=u.second,
            add_keyword={'HIERARCH GIULIA DARK SUB': True})
        return ccd

    def _correctForFlat(self, ccd):
        mFlat = self.getFlatImage()
        print('Correcting for flat')
        ccd = ccdproc.flat_correct(ccd, mFlat)
        return ccd

    def _makeClippedCombiner(self, ccdData_list, n=3):
        ccdComb = Combiner(ccdData_list)
        medCcd = ccdComb.median_combine()
        minclip = np.median(medCcd.data) - n*medCcd.data.std()
        maxclip = np.median(medCcd.data) + n*medCcd.data.std()
        ccdComb.minmax_clipping(min_clip=minclip, max_clip=maxclip)
        return ccdComb


#     def _iterativeSigmaClipping(self, combiner, n):
#         old = 0
#         new = combiner.data_arr.mask.sum()
#         print("old %d" % old)
#         print("new %d" % new)
#         while(new>old):
#             combiner.sigma_clipping(low_thresh=n, high_thresh=n,
#                                     func=np.ma.median, dev_func=np.ma.std)
#             old = new
#             new = combiner.data_arr.mask.sum()
#             print("old %d" % old)
#             print("new %d" % new)

    def _iterativeSigmaClipping(self, combiner, n):
        old = 0
        new = combiner.data_arr.mask.sum()
        while(new>old):
            combiner.sigma_clipping(low_thresh=n, high_thresh=n,
                                    func=np.ma.median, dev_func=np.ma.std)
            old = new
            new = combiner.data_arr.mask.sum()

    def _overscanAndtrim(self, ccd):
        poly_model = models.Polynomial1D(1)
        ccd_OvTrim = []
        if type(ccd) == list:
            for ima in ccd:
                ima = ccdproc.subtract_overscan(ima, overscan=ima[:4, :],
                                                overscan_axis=0, median=True,
                                                model=poly_model)
                ima = ccdproc.subtract_overscan(ima, overscan=ima[2044:, :],
                                                overscan_axis=0, median=True,
                                                model=poly_model)
                ima = ccdproc.subtract_overscan(ima, overscan=ima[:, :4],
                                                overscan_axis=1, median=True,
                                                model=poly_model)
                ima = ccdproc.subtract_overscan(ima, overscan=ima[:, 2044:],
                                                overscan_axis=1, median=True,
                                                model=poly_model)
                ima = ccdproc.trim_image(ima, ima.header['DATASEC'])
                ccd_OvTrim.append(ima)
        else:
            ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:4, :],
                                            overscan_axis=0, median=True,
                                            model=poly_model)
            ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[2044:, :],
                                            overscan_axis=0, median=True,
                                            model=poly_model)
            ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:, :4],
                                            overscan_axis=1, median=True,
                                            model=poly_model)
            ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:, 2044:],
                                            overscan_axis=1, median=True,
                                            model=poly_model)
            ccd_OvTrim = ccdproc.trim_image(
                ccd, ccd.header['DATASEC'])
        return ccd_OvTrim

    def _adu2Electron(self, ccd):
        print('Converting from ADU to e-')
        return ccdproc.gain_correct(ccd,
                                    ccd.header['GAIN'], u.electron/u.adu)
