'''
Created on 20 nov 2018

@author: gcarla
'''
import numpy as np
from astropy.table.table import Table
from photutils.datasets.make import make_gaussian_sources_image,\
    apply_poisson_noise, make_noise_image
from photutils.utils.check_random_state import check_random_state



class ImageCreator(object):
    
    def __init__(self,
                 shape,
                 stddevXRange=[2., 3],
                 stddevYRange=[2., 3],
                 fluxInPhotons=[1000., 10000]
                 ):
        self._shape= shape
        self._stddevXRange= stddevXRange
        self._stddevYRange= stddevYRange
        self._fluxInPhotons= fluxInPhotons
        self._ronInPhotons= 0
        self._usePoissonNoise= False
        self._seed=check_random_state(12345)
        self._forceStddevXEqualToStddevY= True
        self._table= None
        

    def useReadOutNoise(self, ronInPhotons):
        self._ronInPhotons= ronInPhotons


    def usePoissonNoise(self, trueOrFalse):
        self._usePoissonNoise= trueOrFalse
        

    def createMultipleGaussian(self,
                               nStars=100):
        xMean= np.random.uniform(1, self._shape[1]-1, nStars)
        yMean= np.random.uniform(1, self._shape[0]-1, nStars)
        sx= np.random.uniform(
            self._stddevXRange[0], self._stddevXRange[1], nStars)
        if self._forceStddevXEqualToStddevY:
            sy= sx
        else:
            sy= np.random.uniform(
                self._stddevYRange[0], self._stddevYRange[1], nStars)
            
        amp= np.random.uniform(
            self._amplitudeInPhotons[0], self._amplitudeInPhotons[1], nStars)
        
        self._table = Table() 
        self._table['x_mean']= xMean
        self._table['y_mean']= yMean
        self._table['x_stddev']= sx
        self._table['y_stddev']= sy
        self._table['amplitude']= amp
        ima= make_gaussian_sources_image(self._shape, self._table) 
        if self._usePoissonNoise: 
            ima= apply_poisson_noise(ima, random_state=self._seed)
        if self._ronInPhotons != 0:
            ron= make_noise_image(
                self._shape, type= 'gaussian', mean= 0, 
                stddev= self._ronInPhotons)
            ima= ima + ron 
        return ima

    
    def getTable(self):
        return self._table
    
    
    def createGaussianImage(self, 
        posX, posY, stdX, stdY, fluxInPhotons):
        
        table = Table()     
        table['x_mean']= [posX]
        table['y_mean']= [posY]
        table['x_stddev']= [stdX]
        table['y_stddev']= [stdY]
        table['flux']= [fluxInPhotons]
        ima= make_gaussian_sources_image(self._shape, table)
        if self._usePoissonNoise: 
            ima= apply_poisson_noise(ima, random_state=self._seed)
        if self._ronInPhotons != 0:
            ron= make_noise_image(
                self._shape, type= 'gaussian', mean= 0, 
                stddev= self._ronInPhotons)
            ima= ima + ron 
        return ima
