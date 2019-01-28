'''
Created on 3 dic 2018

@author: giulia
'''


from ccdproc.image_collection import ImageFileCollection
import numpy as np
import ccdproc
import pickle


class DataReduction():

    FRAMETYPE_SCIENCE= 'SCIENCE'
    FRAMETYPE_SKY= 'SKY'
    FRAMETYPE_FLAT= 'FLAT'
    FRAMETYPE_DARK= 'DARK'

    def __init__(self, data_dir,
                 keys=['FRAMETYP', 'OBJECT', 'DIT', 'FILTER',
                       'OBJRA', 'OBJDEC', 'DATE-OBS', 'DATE']):
        self.data_dir = data_dir
        self.keys = keys
        self._imagesCollection= None
        self._integrationTime= None
        self._filterType= None
        self._objectName= None

        self._dark= None
        self._flat= None
        self._sky= None
        self._scienceFinal= None

    def _createImageCollection(self):
        self._imagesCollection= ImageFileCollection(self.data_dir, self.keys)
        return self._imagesCollection

    def _reset(self):
        self._dark= None
        self._flat= None
        self._sky= None
        self._scienceFinal= None

    def saveToFile(self, objToSave, filename):
        with open(filename, 'wb') as handle:
            pickle.dump(objToSave, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restoreFromFile(filename):
        with open(filename, 'rb') as handle:
            obj= pickle.load(handle)
            return obj

    def setIntegrationTime(self, integrationTime):
        self._integrationTime= integrationTime
        self._reset()

    def setFilterType(self, filterType):
        self._filterType= filterType
        self._reset()

    def setObjectName(self, objectName):
        self._objectName= objectName
        self._reset()

    def getScienceImage(self):
        if self._scienceFinal is None:
            self._computeScienceImage()
        return self._scienceFinal

    def _getSkyImage(self):
        if self._sky is None:
            self._computeSkyImage()
        return self._sky

    def _getFlatFieldImage(self):
        if self._flat is None:
            self._computeFlatFieldImage()
        return self._flat

    def _getDarkImage(self):
        if self._dark is None:
            self._computeDarkImage()
        return self._dark

    def createMask(self, ccddata_list, sigmaClipWidth=3):
        framesCube=np.array([d.data for d in ccddata_list])
        maskStd= self._maskOutliersOnStd(framesCube, sigmaClipWidth)
        maskMedian= self._maskOutliersOnMedian(framesCube, sigmaClipWidth)
        unionOfMasks= np.clip(maskMedian+maskStd, 0, 1)
        return unionOfMasks

    def _maskOutliersOnStd(self, framesCube, sigmaClipWidth=3):
        framesStd= framesCube.std(axis=0)
        loStd= np.median(framesStd) - sigmaClipWidth*framesStd.std()
        hiStd= np.median(framesStd) + sigmaClipWidth*framesStd.std()
        maskStd=np.zeros(framesStd.shape)
        maskStd[np.where(framesStd<loStd)]=1
        maskStd[np.where(framesStd>hiStd)]=1
        return maskStd

        # mask outliers on median (too high or too small)
    def _maskOutliersOnMedian(self, framesCube, sigmaClipWidth=3):
        framesMedian= np.median(framesCube, axis=0)
        loMedian= np.median(framesMedian) - sigmaClipWidth*framesMedian.std()
        hiMedian= np.median(framesMedian) + sigmaClipWidth*framesMedian.std()
        maskMedian=np.zeros(framesMedian.shape)
        maskMedian[np.where(framesMedian>hiMedian)]=1
        maskMedian[np.where(framesMedian<loMedian)]=1
        return maskMedian

    def _computeDarkImage(self, createMask=True):
        self._darkFileList=[]
        self._darkIma=[]
        for dark, fname in self._createImageCollection().hdus(
                FRAMETYP=self.FRAMETYPE_DARK,
                DIT=self._integrationTime,
                return_fname=True):
            meta=dark.header
            meta['filename']=fname
            self._darkIma.append((ccdproc.CCDData(data=dark.data,
                                                  meta=meta, unit="adu")))
            self._darkFileList.append(fname)
        del self._darkIma[20:50]
        del self._darkFileList[20:50]

        self._darkMedian= np.median(np.array([d.data
                                              for d in self._darkIma]), axis=0)

        if createMask is True:
            mask= self.createMask(self._darkIma)
        else:
            mask= np.zeros(self._darkMedian.shape)

        self._dark= np.ma.masked_array(self._darkMedian, mask=mask)

    def _computeFlatFieldImage(self, createMask=True):
        self._flatFileList=[]
        self._flatIma=[]
        for flat, fname in self._createImageCollection().hdus(
                FRAMETYP=self.FRAMETYPE_FLAT,
                FILTER=self._filterType,
                DIT=self._integrationTime,
                return_fname=True):
            meta=flat.header
            meta['filename']= fname
            self._flatIma.append((ccdproc.CCDData(data=flat.data,
                                                  meta=meta,
                                                  unit="adu")))
            self._flatFileList.append(fname)
            # Per 20161019 fai 'del _flatFileList[11:28]', 'del
            # _flatIma[11:28]'
        del self._flatIma[11:28]
        del self._flatFileList[11:28]
        self._flatMedian= np.median(np.array([d.data for d in self._flatIma]),
                                    axis=0)

        if createMask is True:
            mask= self.createMask(self._flatIma)
        else:
            mask= np.zeros(self._flatMedian.shape)
#         self._flatMedianMasked= np.ma.masked_array(self._flatMedian, mask=mask)
#         self._flat_dark= self._flatMedianMasked-self._getDarkImage()
#         self._flat= self._flat_dark/np.ma.median(self._flat_dark)

        flats=[]
        for flat in self._flatIma:
            flatMasked= np.ma.masked_array(flat.data, mask=mask)
            flatDarkSubtracted= flatMasked - self._getDarkImage()
            flats.append(flatDarkSubtracted / np.ma.median(flatDarkSubtracted))
        self._flat= np.ma.median(np.ma.array(flats), axis=0)

    def _computeSkyImage(self):
        self._skyFileList=[]
        self._skyIma=[]
        for sky, fname in self._createImageCollection().hdus(
                FRAMETYP=self.FRAMETYPE_SKY,
                FILTER=self._filterType,
                DIT=self._integrationTime,
                OBJECT=self._objectName,
                return_fname=True):
            meta=sky.header
            meta['filename']= fname
            self._skyIma.append((ccdproc.CCDData(data=sky.data,
                                                 meta=meta,
                                                 unit="adu")))
            self._skyFileList.append(fname)
        'PRIMO DITHERING'
        del self._skyFileList[35::]
        del self._skyIma[35::]

#         self._skyMedian= np.median(np.array([d.data
#                                              for d in self._skyIma]), axis=0)
#         self._sky= np.ma.masked_array(self._skyMedian,
#                                       mask=self.createMask(self._skyIma))
        skys=[]
        for sky in self._skyIma:
            skys.append(
                (sky.data - self._getDarkImage())/self._getFlatFieldImage())
        self._sky= np.ma.median(np.ma.dstack(skys), axis=2)

    def _computeScienceImage(self):
        self._scienceFileList=[]
        self._scienceIma=[]
        for science, fname in self._createImageCollection().hdus(
                FRAMETYP=self.FRAMETYPE_SCIENCE,
                FILTER=self._filterType,
                DIT=self._integrationTime,
                OBJECT=self._objectName,
                return_fname=True):
            meta=science.header
            meta['filename']= fname
            self._scienceIma.append((ccdproc.CCDData(data=science.data,
                                                     meta=meta,
                                                     unit="adu")))
            self._scienceFileList.append(fname)
        # 'PRIMO DITHERING'
        del self._scienceFileList[9::]
        del self._scienceIma[9::]

#         scie= np.median(np.array([a.data for a in self._scienceIma]), axis=0)
#         self._science_sky= scie-self._getSkyImage()
#         self._scienceFinal= self._science_sky/self._getFlatFieldImage()
        scis=[]
        for sci in self._scienceIma:
            scis.append((sci - self._getDarkImage())/self._getFlatFieldImage())
        sci= np.ma.median(np.ma.dstack(scis), axis=2)
        self._scienceFinal = sci - self._getSkyImage()
