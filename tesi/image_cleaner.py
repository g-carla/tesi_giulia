'''
Created on 16 ott 2018

@author: gcarla
'''

import numpy as np
import ccdproc
from astropy import units as u
from ccdproc import CCDData, Combiner
from astropy.modeling import models
from astropy.stats import SigmaClip


class ImageCleaner():
    """
    Reduce astronomical images performing dark subtraction, flatfield
    correction and sky subtraction.
    The reduction can be performed either from scratch, that is computing
    masterdark, masterflat and mastersky from lists of raw images and then
    using them to calibrate the scientific image, either with masterdark,
    masterflat and mastersky given by the user.


    Parameters
    ----------
    sci_imas_raw: list or `~astropy.nddata.CCDData`, optional
            Scientific image to be corrected. It can be a single image as
            CCDData or a list of images as CCDData.

    dark_imas_raw: list or None, optional
            List of dark images as CCDData.
            Default is ``None``.

    flat_imas_raw: list or None, optional
            List of flatfield images as CCDData.
            Default is ``None``.

    sky_imas_raw: list or None, optional
            List of sky images as CCDData.
            Default is ``None``.

    master_dark: `~astropy.nddata.CCDData` or None, optional
            The master dark frame to be subtracted from the CCDData.
            Default is ``None``.

    master_flat: `~astropy.nddata.CCDData` or None, optional
            The master flat frame to be divided into CCDData. It mustn't be
            normalized to unity yet.
            Default is ``None``.

    master_sky: `~astropy.nddata.CCDData` or None, optional
            The master sky frame to be subtracted from the CCDData.
            Default is ``None``.

    bad_pixel_mask: `numpy.ndarray` or None, optional
            The bad pixel mask for the data.
            Default is ``None``.
    """

    def __init__(self,
                 sci_imas_raw,
                 dark_imas_raw=None,
                 flat_imas_raw=None,
                 sky_imas_raw=None,
                 master_dark=None,
                 master_flat=None,
                 master_sky=None,
                 bad_pixel_mask=None):
        self._science = sci_imas_raw
        self._darks = dark_imas_raw
        self._flats = flat_imas_raw
        self._skies = sky_imas_raw
        self.masterDark = master_dark
        self.masterFlat = master_flat
        self.masterSky = master_sky
        self.mask = bad_pixel_mask

    def setScienceImage(self, ima):
        self._science = ima

    def setMasterDark(self, mDark):
        self.masterDark = mDark

    def setMasterFlat(self, mFlat):
        self.masterFlat = mFlat

    def setMasterSky(self, mSky):
        self.masterSky = mSky

    def getScienceImage(self, unit='electron'):
        """
        Return the scientific image after the calibration.

        Parameters
        ----------
        unit: str
            Unit in which the reduced scientific image has to be returned.
            It can be `electron` or `ADU`.
            Default is `electron`.

        Return
        ------
        masterSci: `~astropy.nddata.CCDData`
            Reduced scientific image.
        """

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
        # ATTENZIONE: questo masterflat non e' normalizzato, perche' viene
        # normalizzato in _correctForFlat
        return self.masterFlat

    def getDarkImage(self):
        if self.masterDark is None:
            self._computeDarkImage()
        return self.masterDark

    def getBadPixelMask(self):
        if (self.mask is None and self.masterDark is None):
            self._computeBadPixelMaskUsingDarkFrames()
        elif self.masterDark is not None:
            self.mask = self.masterDark.mask
        return self.mask

    def _computeScienceImage(self):
        print('\n MASTER SCIENCE: \n')
        #        self.sciTrim = self._overscanAndtrim(self.science)
        # TODO: use ccd_process?
        if type(self._science) == list:
            scisCorrected = []
            for sci in self._science:
                darkCorrection = self._subtractDark(sci)
                flatCorrection = self._correctForFlat(darkCorrection)
                skyCorrection = self._subtractSky(flatCorrection)
                scisCorrected.append(skyCorrection)
            print('Sigma clipping...')
            sciCombiner = Combiner(scisCorrected)
            sciCombiner.sigma_clipping(low_thresh=3., high_thresh=3.,
                                       func=np.ma.median, dev_func=np.ma.std)
            print('Median combine...')
            medianSci = sciCombiner.median_combine()
            mask = self.getBadPixelMask() + medianSci.mask
            print('Getting master science frame...')
            self.masterSci = CCDData(
                medianSci,
                mask=mask,
                unit='adu')
            print('Writing the header...')
            self.masterSci.header = self._science[0].meta
            # TODO: risky header?
#             self.masterSci.header['FRAMETYP'] = \
#                 self._science[0].header['FRAMETYP']
#             self.masterSci.header['OBJECT'] = self._science[0].header['OBJECT']
#             self.masterSci.header['DIT'] = self._science[0].header['DIT']
#             self.masterSci.header['FILTER'] = \
#                 self._science[0].header['FILTER']
#             self.masterSci.header['OBJRA'] = self._science[0].header['OBJRA']
#             self.masterSci.header['OBJDEC'] = self._science[0].header['OBJDEC']
#             self.masterSci.header['DATE'] = self._science[0].header['DATE']
#             self.masterSci.header['GAIN'] = self._science[0].header['GAIN']
        else:
            sci_dark = self._subtractDark(self._science)
            sciFlat = self._correctForFlat(sci_dark)
            print('Getting master science frame...')
            self.masterSci = self._subtractSky(sciFlat)
            print('Writing the header...')
            self.masterSci.header = self._science.header

        if self._unit == 'electron':
            self.masterSci = self._adu2Electron(self.masterSci)
            self.masterSci.header['UNIT'] = 'electrons'

    def _computeSkyImage(self):
        print('\n MASTER SKY: \n')
#        self.skyTrim = self._overscanAndtrim(self.skies)
        skiesCorrected = []
        for sky in self._skies:
            skyDark = self._subtractDark(sky)
            skyFlat = self._correctForFlat(skyDark)
            skiesCorrected.append(skyFlat)
        print('Sigma clipping...')
        skyCombiner = Combiner(skiesCorrected)
        skyCombiner.sigma_clipping(low_thresh=3., high_thresh=3.,
                                   func=np.ma.median, dev_func=np.ma.std)
        print('Median combine..')
        medianSky = skyCombiner.median_combine()

        mask = self.getBadPixelMask() + medianSky.mask
        print('Getting master sky frame...')
        self.masterSky = CCDData(medianSky, mask=mask, unit='adu')
        self.masterSky.header = skiesCorrected[0].meta
        # TODO: risky header?
#         self._skyCombiner = Combiner(self._skiesCorrected)
#         self._iterativeSigmaClipping(self._skyCombiner, 1)
#         self.masterSky = self._skyCombiner.median_combine()

    def _computeFlatImage(self):
        print('\n MASTER FLAT: \n')
#       self.flatsTrim = self._overscanAndtrim(self.flats)
        print('Dark subtraction...')
        flatsDarkSubtracted = []
        for ima in self._flats:
            imaDarkSub = self._subtractDark(ima)
            flatsDarkSubtracted.append(imaDarkSub)

        print('Sigma clipping...')
        flatCombiner = Combiner(flatsDarkSubtracted)
        flatCombiner.sigma_clipping(low_thresh=3., high_thresh=3.,
                                    func=np.ma.median, dev_func=np.ma.std)
        print('Median combine...')
        medianFlat = flatCombiner.median_combine()
        mask = self.getBadPixelMask() + medianFlat.mask
        print('Getting master flat frame...')
        self.masterFlat = CCDData(medianFlat, mask=mask, unit='adu')

        print('Writing the master flat\'s header...')
        # TODO: risky header?
        self.masterFlat.header = flatsDarkSubtracted[0].meta

    def _computeBadPixelMaskUsingDarkFrames(self):
        print('Computing the bad pixel mask...')
        sigClip = SigmaClip(sigma=3., cenfunc='median', maxiters=10)
        darkClipped = sigClip(self._darkMedian, axis=0, masked=True)
        self.mask = darkClipped.mask

    def _computeDarkImage(self):
        print('\n MASTER DARK: \n')
#        self.darksTrim = self._overscanAndtrim(self.darks)

        print('Sigma clipping...')
        darksCombiner = Combiner(self._darks)
        darksCombiner.sigma_clipping(low_thresh=3, high_thresh=3,
                                     func=np.ma.median, dev_func=np.ma.std)
        print('Median combine...')
        self._darkMedian = darksCombiner.median_combine()
        self._computeBadPixelMaskUsingDarkFrames()
        mask = self.mask + self._darkMedian.mask
        print('Getting master dark frame...')
        self.masterDark = CCDData(self._darkMedian,
                                  mask=mask, unit='adu')

        print('Writing the master dark\'s header...')
        self.masterDark.header = self._darks[0].meta

#         self.masterDark.header[
#             'DIT'] = self._darkCombiner.ccd_list[0].header['DIT']
#         self.masterDark.header['RDNOISE'] = self._darkCombiner.ccd_list[
#             0].header['RDNOISE']
#         self.masterDark.header[
#             'GAIN'] = self._darkCombiner.ccd_list[0].header['GAIN']
#         self.masterDark.header['DATASEC'] = self._darkCombiner.ccd_list[
#             0].header['DATASEC']

    def _subtractDark(self, ccd):
        print('Subtracting dark...')
        masterDark = self.getDarkImage()
        # TODO: add 'dark_exposure' and 'data_exposure'
        ccd = ccdproc.subtract_dark(
            ccd,
            masterDark,
            exposure_time='DIT',
            exposure_unit=u.second,
            add_keyword={'HIERARCH GIULIA DARK SUBTRACTION': True})
        return ccd

    def _correctForFlat(self, ccd):
        masterFlat = self.getFlatImage()
        print('Correcting for flat...')
        ccd = ccdproc.flat_correct(
            ccd,
            masterFlat,
            min_value=0.1,
            add_keyword={
                'HIERARCH GIULIA FLATFIELD CORRECTION': True})
        return ccd

    def _subtractSky(self, ccd):
        print('Subtracting sky...')
        masterSky = self.getSkyImage()
        ccdSkySub = ccd.subtract(masterSky,
                                 handle_mask='first_found')
        return ccdSkySub

    def _adu2Electron(self, ccd):
        print('Converting from ADU to e-')
        return ccdproc.gain_correct(ccd,
                                    ccd.header['GAIN'],
                                    u.electron / u.adu)

#     def _makeClippedCombiner(self, ccdData_list, n=3):
#         ccdComb = Combiner(ccdData_list)
#         medCcd = ccdComb.median_combine()
#         minclip = np.median(medCcd.data) - n * medCcd.data.std()
#         maxclip = np.median(medCcd.data) + n * medCcd.data.std()
#         ccdComb.minmax_clipping(min_clip=minclip, max_clip=maxclip)
#         return ccdComb

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

#     def _iterativeSigmaClipping(self, combiner, n):
#         old = 0
#         new = combiner.data_arr.mask.sum()
#         while(new > old):
#             combiner.sigma_clipping(low_thresh=n, high_thresh=n,
#                                     func=np.ma.median, dev_func=np.ma.std)
#             old = new
#             new = combiner.data_arr.mask.sum()
#
#     def _overscanAndtrim(self, ccd):
#         poly_model = models.Polynomial1D(1)
#         ccd_OvTrim = []
#         if type(ccd) == list:
#             for ima in ccd:
#                 ima = ccdproc.subtract_overscan(ima, overscan=ima[:4, :],
#                                                 overscan_axis=0, median=True,
#                                                 model=poly_model)
#                 ima = ccdproc.subtract_overscan(ima, overscan=ima[2044:, :],
#                                                 overscan_axis=0, median=True,
#                                                 model=poly_model)
#                 ima = ccdproc.subtract_overscan(ima, overscan=ima[:, :4],
#                                                 overscan_axis=1, median=True,
#                                                 model=poly_model)
#                 ima = ccdproc.subtract_overscan(ima, overscan=ima[:, 2044:],
#                                                 overscan_axis=1, median=True,
#                                                 model=poly_model)
#                 ima = ccdproc.trim_image(ima, ima.header['DATASEC'])
#                 ccd_OvTrim.append(ima)
#         else:
#             ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:4, :],
#                                             overscan_axis=0, median=True,
#                                             model=poly_model)
#             ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[2044:, :],
#                                             overscan_axis=0, median=True,
#                                             model=poly_model)
#             ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:, :4],
#                                             overscan_axis=1, median=True,
#                                             model=poly_model)
#             ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:, 2044:],
#                                             overscan_axis=1, median=True,
#                                             model=poly_model)
#             ccd_OvTrim = ccdproc.trim_image(
#                 ccd, ccd.header['DATASEC'])
#         return ccd_OvTrim
#
#     @staticmethod
#     def convertNanPixels(ima, mask):
#
#         def medianFilteringUsingEightNeighbours(ima, maskIndex):
#             from astropy.nddata.utils import Cutout2D
#             cut = Cutout2D(ima, (maskIndex[1], maskIndex[0]), 3).data
#             cut_mask = Cutout2D(ima, (maskIndex[1], maskIndex[0]), 3).data.mask
#             cut[np.where(cut_mask == True)] = np.ma.median(cut)
#
#         maskIndices = np.argwhere(mask == True)
#         for i in range(maskIndices.shape[0]):
#
#             if (maskIndices[i][0] != 0 and maskIndices[i][0] != 2047 and
#                     maskIndices[i][1] != 0 and maskIndices[i][1] != 2047):
#                 medianFilteringUsingEightNeighbours(ima, maskIndices[i])
#
#             else:
#                 ima[maskIndices[i][0], maskIndices[i][1]] = ima.mean()
#
#         return ima


#         for n in range(maskedIndices.shape[0]):
#             i = maskedIndices[n]
#             if (i[0] != 0 and i[0] != 2047 and i[1] != 0 and i[1] != 2047):
#                 j = np.array([[i[0] + 1, i[1] + 1],
#                               [i[0] + 1, i[1]],
#                               [i[0] + 1, i[1] - 1],
#                               [i[0], i[1] + 1],
#                               [i[0], i[1] - 1],
#                               [i[0] - 1, i[1] - 1],
#                               [i[0] - 1, i[1]],
#                               [i[0] - 1, i[1] + 1]])
#                 yy = j[:, 0]
#                 xx = j[:, 1]
#                 meanOfNeigh = ima[yy, xx].mean()
#                 ima[i[0], i[1]] = meanOfNeigh
#             else:
#                 ima[i[0], i[1]] = ima.mean()
#         return ima
