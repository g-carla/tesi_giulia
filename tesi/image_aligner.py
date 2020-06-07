'''
Created on 08 feb 2019

@author: gcarla
'''
import numpy as np
from skimage.transform._warps import warp
from skimage.transform import ProjectiveTransform
from ccdproc.ccddata import CCDData


class ImageAligner():
    '''
    '''

    def __init__(self,
                 coords_ima1,
                 coords_ima2):
        self._c1 = coords_ima1
        self._c2 = coords_ima2
        self._Nstars = len(coords_ima1)
        self._X1 = np.array([self._c1[i][0] for i in range(self._Nstars)])
        self._Y1 = np.array([self._c1[i][1] for i in range(self._Nstars)])
        self._X2 = np.array([self._c2[i][0] for i in range(self._Nstars)])
        self._Y2 = np.array([self._c2[i][1] for i in range(self._Nstars)])

#     def findTransformationMatrixWithIRAFTables(self):
#         self._X1 = np.array(self._table1['xcentroid'])
#         self._Y1 = np.array(self._table1['ycentroid'])
#         self._X2 = np.array(self._table2['xcentroid'])
#         self._Y2 = np.array(self._table2['ycentroid'])
#         self._Matr1 = np.array([self._X1,
#                                 self._Y1,
#                                 np.ones(len(self._table1))])
#         self._Matr2 = np.array([self._X2,
#                                 self._Y2,
#                                 np.ones(len(self._table2))])
#         self.transfMat = np.dot(self._Matr2, np.linalg.pinv(self._Matr1))
#
#     def findTransformationMatrixWithDAOPHOTTable(self):
#         self._X1 = np.array(self._table1['x_fit'])
#         self._Y1 = np.array(self._table1['y_fit'])
#         self._X2 = np.array(self._table2['x_fit'])
#         self._Y2 = np.array(self._table2['y_fit'])
#         self._Matr1 = np.array([self._X1,
#                                 self._Y1,
#                                 np.ones(len(self._table1))])
#         self._Matr2 = np.array([self._X2,
#                                 self._Y2,
#                                 np.ones(len(self._table2))])
#         self.transfMat = np.dot(self._Matr2, np.linalg.pinv(self._Matr1))

    def findTransformationMatrix(self):
        self._Matr1 = np.array([self._X1,
                                self._Y1,
                                np.ones(self._Nstars)])
        self._Matr2 = np.array([self._X2,
                                self._Y2,
                                np.ones(self._Nstars)])
        self.transfMat = np.dot(self._Matr2, np.linalg.pinv(self._Matr1))

    def getTransformationMatrix(self):
        return self.transfMat

    def getNewCoordinates(self):
        return np.dot(np.linalg.pinv(self.transfMat), self._Matr1)[0:2]

    def applyTransformationOnIma(self, ima2):
        alignedIma = warp(ima2, ProjectiveTransform(matrix=self.transfMat),
                          output_shape=ima2.shape,
                          order=3, mode='constant',
                          cval=np.median(ima2.data))
        return CCDData(alignedIma, unit=ima2.unit)
