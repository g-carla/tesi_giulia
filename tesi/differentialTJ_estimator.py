'''
Created on 02 mar 2019

@author: gcarla
'''

import numpy as np
from tesi import astrometric_error_estimator
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot


class estimateDifferentialTJ():
    '''
    Class to estimate differential tilt jitter

    Parameters
    ----------
    starsTab: list of astropy.table.Table produced from
                a first stellar fit on imasList

    For J 161019 dither 1 x_NGS=417, y_NGS=1376
    '''

    def __init__(self, starsTabs, NGSCoordinates, n=2):
        self.starsTabs = starsTabs
        self.NGSCoordinates = NGSCoordinates
        self._maxShift = n
        self.est = astrometric_error_estimator.EstimateAstrometricError(
            starsTabs)
        self.est.createCubeOfStarsInfo()
        self.starsX = self.est.getStarsPositionX()
        self.starsY = self.est.getStarsPositionY()
        self.fluxes = self.est.getStarsFlux()
        self.nStars= self.starsX.shape[1]
        self.nImages= self.starsX.shape[0]

    def _findIndexTabRelativeToNGS(self):
        xNGS= self.NGSCoordinates[0]
        yNGS= self.NGSCoordinates[1]
        xx = np.array(self.starsTabs[0]['x_fit'])
        yy = np.array(self.starsTabs[0]['y_fit'])
        i = np.argwhere((np.abs(xx-xNGS) < self._maxShift) &
                        (np.abs(yy-yNGS) < self._maxShift))
        return i[0][0]

    def _getNGSCoordinates(self):
        i = self._findIndexTabRelativeToNGS()
        self.NGS_coords = np.vstack((
            np.array([np.array(tab['x_fit'][i]) for tab in self.starsTabs]),
            np.array([np.array(tab['y_fit'][i]) for tab in self.starsTabs]))).T
        return self.NGS_coords

    def _getNGSMeanPosition(self):
        i = self._findIndexTabRelativeToNGS()
        return self.est.getMeanPosition()[:, i]

    def _meanDistanceFromMeanNGS(self):
        return (self.est.getMeanPosition().T-self._getNGSMeanPosition()).T

    def _toPolar(self, cartCoord):
        theta= np.arctan2(cartCoord[1], cartCoord[0])
        rho= np.linalg.norm(cartCoord, axis=0)
        return np.vstack((rho, theta))

    def _makeRotationMatrix(self, theta):
        return np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])

    def getDTJError(self):
        self.polCoord= self._toPolar(self._meanDistanceFromMeanNGS())
        self._allPos= np.dstack((self.est.getStarsPositionX().T,
                                 self.est.getStarsPositionY().T))
        astrometricError= []
        self._distFromMeanNGS = []
        self._rotatedDist = []
        for i in range(self.nStars):
            rotMat= self._makeRotationMatrix(self.polCoord[1, i])
            distFromMeanNGS= self._allPos[i]-self._getNGSMeanPosition()
            rotatedDistance= np.dot(rotMat, distFromMeanNGS.T).T
            astrometricError.append(np.std(rotatedDistance, axis=0))
            self._distFromMeanNGS.append(distFromMeanNGS)
            self._rotatedDist.append(rotatedDistance)
        self._distFromMeanNGS = np.array(self._distFromMeanNGS)
        self._rotatedDist = np.array(self._rotatedDist)
        return np.array(astrometricError)

    def plotDTJError(self, unit='arcsec', leg='yes'):
        ae = self.getDTJError()
        if unit=='arcsec':
            plot(self.polCoord[0]*0.119, ae[:, 0]*0.119*1e03,
                 '.', label="$\sigma_{\parallel}$")
            plot(self.polCoord[0]*0.119, ae[:, 1]*0.119*1e03,
                 '.', label="$\sigma_{\perp}$")
            plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
            plt.ylabel('$\sigma_{tilt\:jitter}$ [mas]', size=12)
        elif unit=='px':
            plot(self.polCoord[0], ae[:, 0], '.', label="$\sigma_{\parallel}$")
            plot(self.polCoord[0], ae[:, 1], '.', label="$\sigma_{\perp}$")
            plt.xlabel('d$_{NGS}$ [px]', size=12)
            plt.ylabel('$\sigma_{tilt\:jitter}$ [px]', size=12)
        if leg == 'yes':
            plt.legend()
        plt.xticks(size=11)
        plt.yticks(size=11)
#        plt.ylim(0, 39)
