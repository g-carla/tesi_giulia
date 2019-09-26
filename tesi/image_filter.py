'''
Created on 6 set 2019

@author: giuliacarla
'''

import ccdproc
from ccdproc.image_collection import ImageFileCollection


class ImageFilter():
    '''
    Select images from a directory depending on the filtering keys and create
    a list of the selection as CCDData.

    Parameters
    ----------
    imasDirectory: path of the directory containing the files to be filtered.
        Files must be in FITS format.
    keys: FITS header's keywords that the user want to use for the filtering.
    '''

    def __init__(self, imasDirectory,
                 keys=['FRAMETYP', 'OBJECT', 'DIT', 'FILTER',
                       'OBJRA', 'OBJDEC', 'DATE-OBS', 'DATE']):
        self._dir = imasDirectory
        self._keys = keys
        self._imagesCollection = None

    def getLists(self, frametyp, intTime, **kwargs):
        '''
        Return a list of the selected images and a list of the corresponding
        filenames.

        Parameters
        ----------
        frametyp: value of the header's keyword relative to the type of image
            that the user want to select.
            Example: frametyp='DARK' for dark frames
                     frametyp='SCIENCE' for frame with the scientific target,
                     etc.
        intTime: value of the header's keyword relative to the exposure time
            of images that the user want to select.
        **kwargs: value of the other keywords available for the filtering.

        Returns
        -------
        imaList: list of selected images as CCDData.
        fileNameList: list of filenames relative to the selected images.
        '''

        self._createLists(frametyp, intTime, **kwargs)
        return self.imaList, self.fileNameList

    def _createImageCollection(self):
        self._imagesCollection = ImageFileCollection(self._dir, self._keys)

    def _getImageCollection(self):
        if self._imagesCollection is None:
            self._createImageCollection()
        return self._imagesCollection

    def _createLists(self, frameType, intTime, **kwargs):
        self.imaList = []
        self.fileNameList = []
        for ima, fname in self._getImageCollection().hdus(
                FRAMETYP=frameType,
                DIT=intTime,
                return_fname=True,
                **kwargs):
            meta=ima.header
            meta['filename']= fname
            self.imaList.append((ccdproc.CCDData(data=ima.data,
                                                 meta=meta,
                                                 unit="adu")))
            self.fileNameList.append(fname)
