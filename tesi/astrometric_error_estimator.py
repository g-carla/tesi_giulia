'''
Created on 12 feb 2019

@author: gcarla
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import clf


class EstimateAstrometricError():
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

    def plotAstroErrorOntheField(self, n=500, fieldUnit='arcsec'):
        err = self.getAstrometricError()
        area = n*err
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

    def plotDisplacements(self, dx, dy, n=300, fieldUnit='arcsec', color='r'):
        xMean = self.getStarsMeanPosition()[0, :]
        yMean = self.getStarsMeanPosition()[1, :]
        fig, ax = plt.subplots()
        if fieldUnit=='arcsec':
            xMeanNew = (xMean-1024)*self.luci1Pxscale
            yMeanNew = (yMean-1024)*self.luci1Pxscale
            dxNew = dx*self.luci1Pxscale*1e03
            dyNew = dy*self.luci1Pxscale*1e03
            ax.set_xlim(-120, 120)
            ax.set_ylim(-120, 120)
            ax.set_xlabel('arcsec', size=13)
            ax.set_ylabel('arcsec', size=13)
            arrowLegendLenght = 10
            legendLabel = '10 mas'
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
            colors = np.hypot(dxNew, dyNew)
            # , width=w)
            plt.quiver(xMeanNew, yMeanNew, dxNew, dyNew, colors)
            cb = plt.colorbar()
            cb.set_label(cbarLabel, rotation=90, size=10)
        else:
            q = ax.quiver(xMeanNew, yMeanNew, dxNew, dyNew, color=color)
            ax.quiverkey(q, 0.4, 1.05, arrowLegendLenght, label=legendLabel,
                         labelpos='E', fontproperties={'size': 12})

#             plt.plot(xMeanNew, yMeanNew,
#                      '.', color='r', markersize=0.1)
#             plt.xlim(-1024, 1024)
#             plt.ylim(-1024, 1024)
#             plt.xlabel('pixel', size=13)
#             plt.ylabel('pixel', size=13)
#             for i in range(len(dx)):
#                 plt.arrow(x=xMeanNew[i], y=yMeanNew[i],
#                           dx=n*dx[i], dy=n*dy[i],
#                           head_width=20, head_length=20, color='r')
        plt.xticks(size=12)
        plt.yticks(size=12)


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
