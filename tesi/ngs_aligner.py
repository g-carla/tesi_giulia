'''
Created on 24 mar 2019

@author: gcarla
'''
import numpy as np
from tesi import astrometric_error_estimator
from astropy.table.table import Table


class alignNGS():

    def __init__(self, starsTabsList, NGSRefCoords, n=2):
        self._starsTabs = starsTabsList
        self._NGSRefCoords = NGSRefCoords
        self._maxShift = n
        self._est = astrometric_error_estimator.EstimateAstrometricError(
            self._starsTabs)
        self._est.createCubeOfStarsInfo()

    def _findIndexTabRelativeToNGS(self):
        xNGS= self._NGSRefCoords[0]
        yNGS= self._NGSRefCoords[1]
        xx = np.array(self._starsTabs[0]['x_fit'])
        yy = np.array(self._starsTabs[0]['y_fit'])
        i = np.argwhere((np.abs(xx-xNGS) < self._maxShift) &
                        (np.abs(yy-yNGS) < self._maxShift))
        return i[0][0]

    def _getNGSCoordinates(self):
        i = self._findIndexTabRelativeToNGS()
        self.allNGSCoords = np.vstack((
            np.array([np.array(tab['x_fit'][i]) for tab in self._starsTabs]),
            np.array([np.array(tab['y_fit'][i]) for tab in self._starsTabs]))).T
        return self.allNGSCoords

    def _getNGSMeanPosition(self):
        i = self._findIndexTabRelativeToNGS()
        return self._est.getMeanPosition()[:, i]

    def _getDisplacementsFromMeanNGS(self):
        self.NGS_disp = self._getNGSCoordinates()-self._getNGSMeanPosition()
        return self.NGS_disp

    def alignCoordsOnMeanNGS(self):
        disp = self._getDisplacementsFromMeanNGS()
        dx = disp[:, 0]
        dy = disp[:, 1]
        self.starsTabsNew = []
        for i in range(len(self._starsTabs)):
            tab = Table()
            tab['x_fit'] = self._starsTabs[i]['x_fit'] - dx[i]
            tab['y_fit'] = self._starsTabs[i]['y_fit'] - dy[i]
            tab['flux_fit'] = self._starsTabs[i]['flux_fit']
            self.starsTabsNew.append(tab)

    def getNewStarsTabsList(self):
        return self.starsTabsNew
