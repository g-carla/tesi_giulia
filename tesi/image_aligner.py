'''
Created on 08 feb 2019

@author: gcarla
'''
import numpy as np
from skimage.transform._warps import warp
from skimage.transform import ProjectiveTransform


class ImageAligner():
    '''
    '''

    def __init__(self,
                 tab1,
                 tab2):
        self._table1 = tab1
        self._table2 = tab2

    def setTable1(self, tab1):
        self._table1 = tab1

    def setTable2(self, tab2):
        self._table2 = tab2

    def findTransformationMatrixWithIRAFTables(self):
        self._X1 = np.array(self._table1['xcentroid'])
        self._Y1 = np.array(self._table1['ycentroid'])
        self._X2 = np.array(self._table2['xcentroid'])
        self._Y2 = np.array(self._table2['ycentroid'])
        self._Matr1 = np.array([self._X1,
                                self._Y1,
                                np.ones(len(self._table1))])
        self._Matr2 = np.array([self._X2,
                                self._Y2,
                                np.ones(len(self._table2))])
        self.transfMat = np.dot(self._Matr2, np.linalg.pinv(self._Matr1))

    def findTransformationMatrixWithDAOPHOTTable(self):
        self._X1 = np.array(self._table1['x_fit'])
        self._Y1 = np.array(self._table1['y_fit'])
        self._X2 = np.array(self._table2['x_fit'])
        self._Y2 = np.array(self._table2['y_fit'])
        self._Matr1 = np.array([self._X1,
                                self._Y1,
                                np.ones(len(self._table1))])
        self._Matr2 = np.array([self._X2,
                                self._Y2,
                                np.ones(len(self._table2))])
        self.transfMat = np.dot(self._Matr2, np.linalg.pinv(self._Matr1))

    def getTransformationMatrix(self):
        return self.transfMat

    def getNewCoordinates(self):
        return np.dot(np.linalg.pinv(self.transfMat), self._Matr1)[0:2]

    def applyTransformationOnIma(self, imaTab2):
        alignedIma = warp(imaTab2, ProjectiveTransform(matrix=self.transfMat),
                          output_shape=imaTab2.shape,
                          order=3, mode='constant',
                          cval=np.median(imaTab2.data))
        return alignedIma

    # TODO: alignedIma is numpy.ndarray. Return as CCDData?
