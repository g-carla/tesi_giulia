'''
Created on 16 ott 2018

@author: gcarla
'''
import numpy as np
from ccdproc.combiner import Combiner
import ccdproc
from astropy import units as u
from astropy.modeling import models


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
        self.masterFlatInElectrons = None
        self.masterSky = None
        self.scienceFinal = None

    def _getDarkImage(self):
        if self.masterDark is None:
            self._makeMasterDark()
        return self.masterDark

    def _getFlatImageInElectrons(self):
        if self.masterFlatInElectrons is None:
            self._makeMasterFlat()
        return self.masterFlatInElectrons

    def _getSkyImage(self):
        if self.masterSky is None:
            self._makeMasterSky()
        return self.masterSky

    def getScienceImage(self):
        if self.scienceFinal is None:
            self._calibrateScienceImage()
        return self.scienceFinal

    def _overscanAndtrim(self, ccd_list):
        poly_model = models.Polynomial1D(1)
        ccd_list2 = []
        for ccd in ccd_list:
            ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:4, :],
                                            overscan_axis=0, median=True,
                                            model=poly_model)
            ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[2044:, :],
                                            overscan_axis=0, median=True,
                                            model=poly_model)
            ccd=ccdproc.subtract_overscan(ccd, overscan=ccd[:, :4],
                                          overscan_axis=1, median=True,
                                          model=poly_model)
            ccd = ccdproc.subtract_overscan(ccd, overscan=ccd[:, 2044:],
                                            overscan_axis=1, median=True,
                                            model=poly_model)
            ccd = ccdproc.trim_image(ccd, ccd.header['DATASEC'])
            ccd_list2.append(ccd)
        return ccd_list2

    def _adu2Electron(self, ccd):
        return ccdproc.gain_correct(ccd, ccd.header['GAIN'], u.electron/u.adu)

#     def _scalingFunc(self, arr):
#         return 1./np.ma.average(arr)

    def _makeClippedCombiner(self, ccdData_list, n=3):
        ccdComb = Combiner(ccdData_list)
        medCcd = ccdComb.median_combine()
        minclip = np.median(medCcd.data) - n*medCcd.data.std()
        maxclip = np.median(medCcd.data) + n*medCcd.data.std()
        ccdComb.minmax_clipping(min_clip=minclip, max_clip=maxclip)
        return ccdComb

    def _makeMasterDark(self):
        #self.darksTrim = self._overscanAndtrim(self.darks)
        #darkCombiner = _self._makeClippedCombiner(self.darksTrim)
        darkCombiner = self._makeClippedCombiner(self.darks)
        self.masterDark = darkCombiner.median_combine()
        self.masterDark.header['DIT'] = darkCombiner.ccd_list[0].header['DIT']
        self.masterDark.header['RDNOISE'] = darkCombiner.ccd_list[
            0].header['RDNOISE']
        self.masterDark.header[
            'GAIN'] = darkCombiner.ccd_list[0].header['GAIN']
        self.masterDark.header['DATASEC'] = darkCombiner.ccd_list[
            0].header['DATASEC']
        # TODO: something else to be added to the masterDark.header?

    def _subtractDark(self, ccd):
        ccd = ccdproc.subtract_dark(ccd, self._getDarkImage(),
                                    exposure_time='DIT',
                                    exposure_unit=u.second)
        # add_keyword={'HIERARCH GIULIA DARK SUB': True})
        return ccd

    def _makeMasterFlat(self):
        #self.flatsTrim = self._overscanAndtrim(self.flats)
        self.flatsDarkSubtracted=[]
        for flat in self.flats:
            flat= self._subtractDark(flat)
            self.flatsDarkSubtracted.append(flat)
        flatCombiner= self._makeClippedCombiner(self.flatsDarkSubtracted)
#         flatCombiner= Combiner(flatsDarkSubtracted)
#         flatCombiner.sigma_clipping(low_thresh=3, high_thresh=3,
#                                     func=np.ma.median, dev_func=np.ma.std)
        scalingFunc = lambda arr: 1/np.ma.average(arr)
        flatCombiner.scaling= scalingFunc
        self.masterFlat= flatCombiner.median_combine()
        self.masterFlat.header= self.flatsDarkSubtracted[0].meta
        self.masterFlatInElectrons = self._adu2Electron(self.masterFlat)

        if self.masterFlatInElectrons.data.min() == 0:
            i = np.argwhere(self.masterFlatInElectrons.data==0)
            y = i[:, 0]
            x = i[:, 1]
            self.masterFlatInElectrons.data[y, x] = 2
        elif self.masterFlatInElectrons.data.min() < 0:
            i = np.argwhere(self.masterFlatInElectrons.data==0)
            y = i[:, 0]
            x = i[:, 1]
            self.masterFlatInElectrons.data[y, x] = 2

    def _subtractDarkAndCorrectForFlat(self, ccd_list):
        ccdCorrect_list = []
        for ccd in ccd_list:
            ccd_dark = self._subtractDark(ccd)
            ccdInElectrons = self._adu2Electron(ccd_dark)
            ccdFlat = ccdproc.flat_correct(ccdInElectrons,
                                           self._getFlatImageInElectrons())
            ccdCorrect_list.append(ccdFlat)
        return ccdCorrect_list

    def _makeMasterSky(self):
        #self.skyTrim = self._overscanAndtrim(self.sky)
        self.skiesClean = self._subtractDarkAndCorrectForFlat(self.skies)
        skyCombiner = self._makeClippedCombiner(self.skiesClean, n=0.1)
        self.masterSky = skyCombiner.median_combine()

    def _calibrateScienceImage(self):
        #self.dataTrim = self._overscanAndtrim(self.data)
        if type(self.science) == list:
            self.science = self._subtractDarkAndCorrectForFlat(self.science)
            sciCombiner = Combiner(self.science)
            sciMedian = sciCombiner.median_combine()
            self.scienceFinal = sciMedian.subtract(self._getSkyImage())
            self.scienceFinal.header['DIT'] = self.science[0].header['DIT']
            self.scienceFinal.header[
                'FILTER'] = self.science[0].header['FILTER']
            self.scienceFinal.header['DATE'] = self.science[0].header['DATE']
        else:
            # add_keyword={'HIERARCH GIULIA DARK SUB': True})
            self.sci_dark = self._subtractDark(self.science)
            self.sciInElectrons = self._adu2Electron(self.sci_dark)
            self.sciFlat = ccdproc.flat_correct(self.sciInElectrons,
                                                self._getFlatImageInElectrons())
            self.scienceFinal = self.sciFlat.subtract(self._getSkyImage())
            self.scienceFinal.header = self.science.header
