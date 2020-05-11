'''
Created on 6 set 2019

@author: giuliacarla
'''

import ccdproc
from ccdproc.image_collection import ImageFileCollection


class ImageFilter():
    """
    Select images from a directory depending on the filtering keys entered by
    the user and create a list of the selection as CCDData.

    Parameters
    ----------
    files_path: str
            path of the directory containing the files to be filtered.
            Files must be in FITS format.
    keys: list
        FITS header's keywords that the user want to use for the
        filtering.

    Example
    -------
    path = '/Users/giuliacarla/Documents/INAF/Lavoro/ARGOS/20161019'
    keys = ['FRAMETYP', 'OBJECT', 'DIT', 'FILTER', 'OBJRA', 'OBJDEC',
            'DATE-OBS', 'DATE']
    ima_filter = ImageFilter(path, keys)
    """

    def __init__(self, files_path, keys):
        self._path = files_path
        self._keys = keys
        self._images_collection = None

    def getLists(self, **kwargs):
        """
        Return a list of the selected images and a list of the corresponding
        filenames.

        Parameters
        ----------
        **kwargs: keyword and its value which we want to use for the filtering.

        Returns
        -------
        imas_list: list
                list of selected images as CCDData.
        names_list: list
                list of the filenames corresponding to the selected images.

        Example
        ------- 
        ngc_imas, ngc_names = ima_filter.getLists(FRAMETYP='SCIENCE',
                                                  OBJECT='NGC2419'
                                                  FILTER='J',
                                                  DIT=3.)
        """

        self._createLists(**kwargs)
        return self.imas_list, self.names_list

    def _createImageCollection(self):
        self._images_collection = ImageFileCollection(self._path, self._keys)

    def _getImageCollection(self):
        if self._images_collection is None:
            self._createImageCollection()
        return self._images_collection

    def _createLists(self, **kwargs):
        self.imas_list = []
        self.names_list = []
        for ima, fname in self._getImageCollection().hdus(
                return_fname=True,
                **kwargs):
            meta = ima.header
            meta['filename'] = fname
            self.imas_list.append((ccdproc.CCDData(data=ima.data,
                                                   meta=meta,
                                                   unit="adu")))
            self.names_list.append(fname)
