'''
Created on 12 feb 2019

@author: gcarla
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import clf


class EstimateAstrometricError(object):
    '''
    '''

    def __init__(self,
                 tabsList,
                 unit='mas'):
        self.tabsList = tabsList
        self._unit = unit
        self.starsInfo = self.paramStack = np.dstack((
            np.array([np.array(tab['x_fit']) for tab in self.tabsList]),
            np.array([np.array(tab['y_fit']) for tab in self.tabsList]),
            np.array([np.array(tab['flux_fit']) for tab in self.tabsList])))
        self.starsX = self.paramStack[:, :, 0]
        self.starsY = self.paramStack[:, :, 1]
        self.starsFlux = self.paramStack[:, :, 2]
        self.luci1Pxscale = 0.119

    def setErrorUnit(self, unit):
        self._unit=unit

    def getStarsMeanPosition(self):
        return np.vstack((self.starsX.mean(axis=0),
                          self.starsY.mean(axis=0)))

    def getStarsFlux(self):
        return self.starsFlux

    def getStdX(self):
        '''
        numpy array (M,)
        '''
        if self._unit=='mas':
            return np.std(self.starsX, axis=0)*self.luci1Pxscale
        elif self._unit=='pixel':
            return np.std(self.starsX, axis=0)
        else:
            print('Unknown unit')

    def getStdY(self):
        '''
        numpy array (M,)
        '''
        if self._unit=='mas':
            return np.std(self.starsY, axis=0)*self.luci1Pxscale
        elif self._unit=='pixel':
            return np.std(self.starsY, axis=0)
        else:
            print('Unknown unit')

    def getAstrometricError(self):
        sx = self.getStdX()
        sy = self.getStdY()
        err = np.sqrt(sx**2 + sy**2)
        return err

    def plotStdX(self):
        clf()
        plt.plot(self.getStdX())
        plt.xlabel('N')
        if self._unit=='mas':
            plt.ylabel('$\sigma_{x}$ [mas]')
        elif self._unit=='pixel':
            plt.ylabel('$\sigma_{x}$ [px]')

    def plotStdY(self):
        clf()
        plt.plot(self.getStdY())
        plt.xlabel('N')
        if self._unit=='mas':
            plt.ylabel('$\sigma_{y}$ [mas]')
        elif self._unit=='pixel':
            plt.ylabel('$\sigma_{y}$ [px]')

    def plotAstrometricError(self):
        clf()
        plt.plot(self.getAstrometricError())
        plt.xlabel('N')
        if self._unit=='mas':
            plt.ylabel('$\sigma_{y}$ [mas]')
        elif self._unit=='pixel':
            plt.ylabel('$\sigma_{y}$ [px]')

    def plotAstroErrorOntheField(self, area=40, fieldUnit='arcsec'):
        err = self.getAstrometricError()
        colors = err*1e03
        if fieldUnit=='arcsec':
            xMean = (self.getStarsMeanPosition()[0, :]-1024)*self.luci1Pxscale
            yMean = (self.getStarsMeanPosition()[1, :]-1024)*self.luci1Pxscale
            plt.scatter(xMean, yMean, s=area, c=colors)
            plt.xlim(-120, 120)
            plt.ylim(-120, 120)
            plt.xlabel('arcsec', size=13)
            plt.ylabel('arcsec', size=13)
        elif fieldUnit=='pixel':
            xMean = self.getStarsMeanPosition()[0, :]-1024
            yMean = self.getStarsMeanPosition()[1, :]-1024
            plt.scatter(xMean, yMean, s=area, c=colors)
            plt.xlim(-1024, 1024)
            plt.ylim(-1024, 1024)
            plt.xlabel('pixel', size=13)
            plt.ylabel('pixel', size=13)
        plt.xticks(size=12)
        plt.yticks(size=12)
        cb = plt.colorbar()
        cb.set_label(label='Errore astrometrico [mas]',
                     size=12)
        cb.ax.tick_params(labelsize=11)

    def getDisplacementsFromMeanPositions(self, i):
        posMean = np.vstack([self.getStarsMeanPosition()[0, :],
                             self.getStarsMeanPosition()[1, :]]).T
        posIma = np.vstack((np.array([np.array(self.tabsList[i]['x_fit'])]),
                            np.array([np.array(self.tabsList[i]['y_fit'])]))).T
        d = posMean-posIma
        dxInPixel = d[:, 0]
        dyInPixel = d[:, 1]
        return dxInPixel, dyInPixel

    def plotDisplacements(self, dx, dy, fieldUnit='arcsec', scale=1,
                          color='r'):
        xMean = self.getStarsMeanPosition()[0, :]
        yMean = self.getStarsMeanPosition()[1, :]
        fig, ax = plt.subplots()
        if fieldUnit=='arcsec':
            xMeanNew = (xMean-1024)*self.luci1Pxscale
            yMeanNew = (yMean-1024)*self.luci1Pxscale
            dxNew = dx*self.luci1Pxscale
            dyNew = dy*self.luci1Pxscale
            ax.set_xlim(-120, 120)
            ax.set_ylim(-120, 120)
            ax.set_xlabel('arcsec', size=13)
            ax.set_ylabel('arcsec', size=13)
            arrowLegendLenght = 0.01
            legendLabel = '10 mas'  # 0.01 arcsec = 10 mas
            cbarLabel = '$\Delta$ [mas]'
        elif fieldUnit=='pixel':
            xMeanNew = (xMean-1024)
            yMeanNew = (yMean-1024)
            dxNew = dx
            dyNew = dy
            ax.set_xlim(-1024, 1024)
            ax.set_ylim(-1024, 1024)
            ax.set_xlabel('pixel', size=13)
            ax.set_ylabel('pixel', size=13)
            arrowLegendLenght = 0.1
            legendLabel = '0.1 pixel'
            cbarLabel = '$\Delta$ [px]'
        if color=='multi':
            colors = np.hypot(dxNew, dyNew)*1e03
            # , width=w)
            plt.quiver(xMeanNew, yMeanNew, dxNew/np.sqrt(dxNew**2 + dyNew**2),
                       dyNew/np.sqrt(dxNew**2 + dyNew**2), colors,
                       angles='xy', scale_units='xy', scale=scale)
            cb = plt.colorbar()
            cb.set_label(cbarLabel, rotation=90, size=12)
            cb.ax.tick_params(labelsize=11)
        else:
            q = ax.quiver(xMeanNew, yMeanNew, dxNew, dyNew, color=color,
                          angles='xy', scale_units='xy', scale=scale)
            ax.quiverkey(q, 0.4, 1.05, arrowLegendLenght, label=legendLabel,
                         labelpos='E', fontproperties={'size': 12})
        ax.tick_params(labelsize=12)


#     def plotDisplacementsMinusTT(self, d_x, d_y):
#         xMean = self.getMeanPositionX()
#         yMean = self.getMeanPositionY()
#         plt.plot(xMean, yMean, '.', color='r', markersize=0.1)
#         plt.xlim(0, 2048)
#         plt.ylim(0, 2048)
#         for i in range(len(d_x)):
#             plt.arrow(x=xMean[i], y=yMean[i], dx=300*(d_x[i]-d_x.mean()),
#                       dy=300*(d_y[i]-d_y.mean()), head_width=20,
#                       head_length=20, color='r')


class EstimateDifferentialTiltJitter(EstimateAstrometricError):

    def __init__(self, tabsList, NGSCoordinates, unit='mas', n=2):

        super().__init__(tabsList, unit)
        self.NGSCoordinates = NGSCoordinates
        '''
        For J 161019 dither 1 x_NGS=417, y_NGS=1376
        '''
        self._maxShift = n
        self.nStars= self.starsX.shape[1]
        self.nImages= self.starsX.shape[0]

    def _findIndexTabRelativeToNGS(self):
        xNGS= self.NGSCoordinates[0]
        yNGS= self.NGSCoordinates[1]
        xx = np.array(self.tabsList[0]['x_fit'])
        yy = np.array(self.tabsList[0]['y_fit'])
        i = np.argwhere((np.abs(xx-xNGS) < self._maxShift) &
                        (np.abs(yy-yNGS) < self._maxShift))
        return i[0][0]

#     def _getNGSCoordinates(self):
#         i = self._findIndexTabRelativeToNGS()
#         self.NGS_coords = np.vstack((
#             np.array([np.array(tab['x_fit'][i]) for tab in self.tabsList]),
#             np.array([np.array(tab['y_fit'][i]) for tab in self.tabsList]))).T
#         return self.NGS_coords

    def _getNGSMeanPosition(self):
        i = self._findIndexTabRelativeToNGS()
        return self.getStarsMeanPosition()[:, i]

    def _meanDistanceFromMeanNGS(self):
        return (self.getStarsMeanPosition().T-self._getNGSMeanPosition()).T

    def _toPolar(self, cartCoord):
        theta= np.arctan2(cartCoord[1], cartCoord[0])
        rho= np.linalg.norm(cartCoord, axis=0)
        return np.vstack((rho, theta))

    def _makeRotationMatrix(self, theta):
        return np.array([[np.cos(theta), np.sin(theta)],
                         [-np.sin(theta), np.cos(theta)]])

    def getDTJError(self):
        self.polCoord= self._toPolar(self._meanDistanceFromMeanNGS())
        self._allPos= np.dstack((self.starsX.T,
                                 self.starsY.T))
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
        # plt.figure()
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ae = self.getDTJError()
        if unit=='arcsec':
            plt.plot(self.polCoord[0]*0.119, ae[:, 0]*0.119*1e03,
                     '.', color=cycle[0], label="$\sigma_{\parallel}$")
            plt.plot(self.polCoord[0]*0.119, ae[:, 1]*0.119*1e03,
                     '.', color=cycle[1], label="$\sigma_{\perp}$")
            plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
            plt.ylabel('$\sigma_{tilt\:jitter}$ [mas]', size=12)
        elif unit=='px':
            plt.plot(
                self.polCoord[0], ae[:, 0],
                '.', color=cycle[0], label="$\sigma_{\parallel}$")
            plt.plot(self.polCoord[0], ae[:, 1],
                     '.', color=cycle[1], label="$\sigma_{\perp}$")
            plt.xlabel('d$_{NGS}$ [px]', size=12)
            plt.ylabel('$\sigma_{tilt\:jitter}$ [px]', size=12)
        if leg == 'yes':
            plt.legend()
        plt.xticks(size=11)
        plt.yticks(size=11)
        plt.xlim(0, 240)
        plt.ylim(0, 39)
