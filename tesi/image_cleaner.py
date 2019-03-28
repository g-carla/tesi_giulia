'''
Created on 16 ott 2018

@author: gcarla
'''
import numpy as np
from ccdproc.combiner import Combiner
import ccdproc
from astropy import units as u
from astropy.modeling import models
from ccdproc.core import cosmicray_median, cosmicray_lacosmic


class ImageCleaner():
    '''
    '''

    def __init__(self,
                 darks,
                 flats,
                 skies,
                 science):
        '''
        darks, flats, skies = lists of CCDData
        dataToCalibrate = list of CCDData or single CCDData
        '''
        self.darks = darks
        self.flats = flats
        self.skies = skies
        self.science = science
        self.masterDark = None
        self.masterFlat = None
        self.masterSky = None
        self.sciFinalInADU = None
        self.sciFinalInElectrons = None

    def _setScienceImage(self, sciIma):
        self.science = sciIma

    def getScienceImageInADU(self):
        if self.sciFinalInADU is None:
            self._calibrateScienceImage()
        return self.sciFinalInADU

    def getScienceImageInElectrons(self):
        if self.sciFinalInElectrons is None:
            self.sciFinalInElectrons = self._adu2Electron(
                self.getScienceImageInADU())
        return self.sciFinalInElectrons

    def _getDarkImage(self):
        if self.masterDark is None:
            self._makeMasterDark()
        return self.masterDark

    def _getFlatImage(self):
        if self.masterFlat is None:
            self._makeMasterFlat()
        return self.masterFlat

    def _getSkyImage(self):
        if self.masterSky is None:
            self._makeMasterSky()
        return self.masterSky

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

    def _subtractDark(self, ccd):
        ccd = ccdproc.subtract_dark(
            ccd,
            self._getDarkImage(),
            exposure_time='DIT',
            exposure_unit=u.second,
            add_keyword={'HIERARCH GIULIA DARK SUB': True})
        return ccd

    def _subtractDarkAndCorrectForFlat(self, ccd_list):
        ccdCorrect_list = []
        for ccd in ccd_list:
            ccd_dark = self._subtractDark(ccd)
            #ccdInElectrons = self._adu2Electron(ccd_dark)
            ccdFlat = ccdproc.flat_correct(ccd_dark,
                                           self._getFlatImage())
            ccdCorrect_list.append(ccdFlat)
        return ccdCorrect_list

    def _adu2Electron(self, ccd):
        return ccdproc.gain_correct(ccd, ccd.header['GAIN'], u.electron/u.adu)

    # TODO: remove cosmic rays?

#     def _scalingFunc(self, arr):
#         return 1./np.ma.average(arr)

    def _iterativeSigmaClipping(self, combiner, n):
        old= 0
        new= combiner.data_arr.mask.sum()
        #print("new %d" % new)
        while(new>old):
            combiner.sigma_clipping(low_thresh=n, high_thresh=n,
                                    func=np.ma.median, dev_func=np.ma.std)
            old= new
            new= combiner.data_arr.mask.sum()
            #print("new %d" % new)

    def _makeClippedCombiner(self, ccdData_list, n=3):
        ccdComb = Combiner(ccdData_list)
        medCcd = ccdComb.median_combine()
        minclip = np.median(medCcd.data) - n*medCcd.data.std()
        maxclip = np.median(medCcd.data) + n*medCcd.data.std()
        ccdComb.minmax_clipping(min_clip=minclip, max_clip=maxclip)
        return ccdComb

    def _makeMasterDark(self):
        #self.darksTrim = self._overscanAndtrim(self.darks)
        self.darkCombiner = self._makeClippedCombiner(self.darks)
        self.masterDark = self.darkCombiner.median_combine()
        self.masterDark.header[
            'DIT'] = self.darkCombiner.ccd_list[0].header['DIT']
        self.masterDark.header['RDNOISE'] = self.darkCombiner.ccd_list[
            0].header['RDNOISE']
        self.masterDark.header[
            'GAIN'] = self.darkCombiner.ccd_list[0].header['GAIN']
        self.masterDark.header['DATASEC'] = self.darkCombiner.ccd_list[
            0].header['DATASEC']
        # TODO: something else to be added to the masterDark.header?

    def _makeMasterFlat(self):
        #self.flatsTrim = self._overscanAndtrim(self.flats)
        self.flatsDarkSubtracted=[]
        for flat in self.flats:
            flat= self._subtractDark(flat)
            self.flatsDarkSubtracted.append(flat)
        self.flatCombiner= self._makeClippedCombiner(self.flatsDarkSubtracted)
#         flatCombiner= Combiner(flatsDarkSubtracted)
#         flatCombiner.sigma_clipping(low_thresh=3, high_thresh=3,
#                                     func=np.ma.median, dev_func=np.ma.std)
        scalingFunc = lambda arr: 1/np.ma.average(arr)
        self.flatCombiner.scaling= scalingFunc
        self.masterFlat= self.flatCombiner.median_combine()
        self.masterFlat.header= self.flatsDarkSubtracted[0].meta
        #self.masterFlatInElectrons = self._adu2Electron(self.masterFlat)

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

    def _makeMasterSky(self):
        #self.skyTrim = self._overscanAndtrim(self.skies)
        self.skiesClean = self._subtractDarkAndCorrectForFlat(self.skies)
        self.skyCombiner = Combiner(self.skiesClean)
        self._iterativeSigmaClipping(self.skyCombiner, 1)
        self.masterSky = self.skyCombiner.median_combine()

    def _calibrateScienceImage(self):
        #self.sciTrim = self._overscanAndtrim(self.science)
        if type(self.science) == list:
            self.sciCombiner = Combiner(
                self._subtractDarkAndCorrectForFlat(self.science))
            self.sciMedian = self.sciCombiner.median_combine()
            self.sciFinalInADU = self.sciMedian.subtract(self._getSkyImage())
            self.sciFinalInADU.header['DIT'] = self.science[0].header['DIT']
            self.sciFinalInADU.header[
                'FILTER'] = self.science[0].header['FILTER']
            self.sciFinalInADU.header['DATE'] = self.science[0].header['DATE']
        else:
            self.sci_dark = self._subtractDark(self.science)
            #self.sciInElectrons = self._adu2Electron(self.sci_dark)
            self.sciFlat = ccdproc.flat_correct(self.sci_dark,
                                                self._getFlatImage())
            self.sciFinalInADU = self.sciFlat.subtract(self._getSkyImage())
            # self.sciFinalInADU = cosmicray_lacosmic(self.sci_sky,
            #                                        sigclip=5)
            self.sciFinalInADU.header = self.science.header
