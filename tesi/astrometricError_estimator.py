'''
Created on 12 feb 2019

@author: gcarla
'''
import numpy as np
from tesi import image_fitter, sandbox
import matplotlib.pyplot as plt
from matplotlib.pyplot import clf, colorbar
from photutils.aperture.circle import CircularAperture


class EstimateAstrometricError():
    '''
    '''

    def __init__(self,
                 tabsList):
        self.tabsList = tabsList

    def createCubeOfStarsInfo(self):
        '''
        Assuming tabsList contains N tables produced in the fitting
        of N images
        Assuming each table contains the fitted parameters of M stars 
        This function selects only stars position and flux and
        returns a cube with shape (N,M,3) 
        '''
        self.paramStack = np.dstack((
            np.array([np.array(tab['x_fit']) for tab in self.tabsList]),
            np.array([np.array(tab['y_fit']) for tab in self.tabsList]),
            np.array([np.array(tab['flux_fit']) for tab in self.tabsList])))

    def getStarsPositionX(self):
        '''
        Returns an array with shape (N,M) with stars xPosition
        '''
        return self.paramStack[:, :, 0]

    def getStarsPositionY(self):
        '''
        Returns an array with shape (N,M) with stars yPosition
        '''
        return self.paramStack[:, :, 1]

    def getStarsFlux(self):
        '''
        Returns an array with shape (N,M) with stars flux
        '''
        return self.paramStack[:, :, 2]

    def getMeanPositionX(self):
        return self.getStarsPositionX().mean(axis=0)

    def getMeanPositionY(self):
        return self.getStarsPositionY().mean(axis=0)

    def getMeanPosition(self):
        return np.vstack((self.getMeanPositionX(), self.getMeanPositionY()))

    def getStdXinPixels(self):
        '''
        numpy array (M,)
        '''
        return np.std(self.getStarsPositionX(), axis=0)

    def getStdXinArcsecs(self, pixelscale=0.119):
        return pixelscale*self.getStdXinPixels()

    def getStdYinPixels(self):
        '''
        numpy array (M,)
        '''
        return np.std(self.getStarsPositionY(), axis=0)

    def getStdYinArcsecs(self, pixelscale=0.119):
        return pixelscale*self.getStdYinPixels()

    def getStandardAstrometricErrorinPixels(self):
        sx = self.getStdXinPixels()
        sy = self.getStdYinPixels()
        err = np.sqrt(sx**2 + sy**2)
        return err

    def getStandardAstrometricErrorinArcsec(self):
        sx = self.getStdXinArcsecs()
        sy = self.getStdYinArcsecs()
        err = np.sqrt(sx**2 + sy**2)
        return err

    def plotStdXinPixels(self):
        clf()
        plt.plot(self.getStdX())
        plt.xlabel('N')
        plt.ylabel('$\sigma_{x}$ [px]')

    def plotStdXinArcsec(self):
        clf()
        plt.plot(self.getStdXinArcsecs())
        plt.xlabel('N')
        plt.ylabel('$\sigma_{x}$ [arcsec]')

    def plotStdYinPixels(self):
        clf()
        plt.plot(self.getStdY())
        plt.xlabel('N')
        plt.ylabel('$\sigma_{y}$ [px]')

    def plotStdYinArcsec(self):
        clf()
        plt.plot(self.getStdYinArcsecs())
        plt.xlabel('N')
        plt.ylabel('$\sigma_{y}$ [arcsec]')

    def plotStandardAstroErrorOntheField(self, n):
        err = self.getStandardAstrometricErrorinArcsec()
        xMean = self.getMeanPositionX()
        yMean = self.getMeanPositionY()
        area = n*err
        colors = err*1e03
        plt.scatter(xMean, yMean, s=area, c=colors)
        plt.xlim(0, 2048)
        plt.ylim(0, 2048)
        cb = plt.colorbar()
        cb.set_label(label='Errore astrometrico [mas]',
                     size=12)
        cb.ax.tick_params(labelsize=11)

    def plotStandardAstroErrorOntheFieldInArcsec(self, n=300):
        err = self.getStandardAstrometricErrorinArcsec()
        xMean = (self.getMeanPositionX()-1024)*0.119
        yMean = (self.getMeanPositionY()-1024)*0.119
        area = n*err
        colors = err*1e03
        #plt.scatter(xMean, yMean, s=area, c=colors)
        plt.scatter(xMean, yMean, c=colors)
        plt.xlim(-120, 120)
        plt.ylim(-120, 120)
        plt.xlabel('arcsec', size=13)
        plt.ylabel('arcsec', size=13)
        plt.xticks(size=12)
        plt.yticks(size=12)
        cb = plt.colorbar()
        cb.set_label(label='Errore astrometrico [mas]',
                     size=12)
        cb.ax.tick_params(labelsize=11)

    def getDisplacementsFromMeanPositions(self, i):
        posMean = np.vstack([self.getMeanPositionX(),
                             self.getMeanPositionY()]).T
        posIma = np.vstack((np.array([np.array(self.tabsList[i]['x_fit'])]),
                            np.array([np.array(self.tabsList[i]['y_fit'])]))).T
        d = posMean-posIma
        d_x = d[:, 0]
        d_y = d[:, 1]
        return d_x, d_y

    def plotDisplacements(self, d_x, d_y, n=300):
        xMean = self.getMeanPositionX()
        yMean = self.getMeanPositionY()
        plt.plot(xMean, yMean, '.', color='r', markersize=0.1)
        plt.xlim(0, 2048)
        plt.ylim(0, 2048)
        for i in range(len(d_x)):
            plt.arrow(x=xMean[i], y=yMean[i], dx=n*d_x[i], dy=n*d_y[i],
                      head_width=20, head_length=20, color='r')

    def plotDisplacementsMinusTT(self, d_x, d_y):
        xMean = self.getMeanPositionX()
        yMean = self.getMeanPositionY()
        plt.plot(xMean, yMean, '.', color='r', markersize=0.1)
        plt.xlim(0, 2048)
        plt.ylim(0, 2048)
        for i in range(len(d_x)):
            plt.arrow(x=xMean[i], y=yMean[i], dx=300*(d_x[i]-d_x.mean()),
                      dy=300*(d_y[i]-d_y.mean()), head_width=20,
                      head_length=20, color='r')
