'''
Created on 20 nov 2018

@author: gcarla
'''
import numpy as np
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from photutils.detection.findstars import IRAFStarFinder
from astropy.modeling import models, fitting



class ImageFitter(object):
    
    def __init__(self,
                 thresholdInPhotons,
                 fwhm):
        self._threshold= thresholdInPhotons
        self._fwhm= fwhm
        
    
    
    def fitSingleStarWithGaussianFit(self, image):
    
        dim_y, dim_x = image.shape
        n = gaussian_fwhm_to_sigma
        self._y, self._x = np.mgrid[:dim_y,:dim_x] 
        
        iraffind = IRAFStarFinder(threshold=self._threshold, fwhm=self._fwhm)
        fit_init_par = iraffind.find_stars(image)
        
        if len(fit_init_par) == 0:
            raise Exception("No star found - (add info please)")
        elif len(fit_init_par) > 1:
            fit_init_par= fit_init_par[0]
    
        fit_model = models.Gaussian2D(x_mean = fit_init_par['xcentroid'], \
                                      y_mean = fit_init_par['ycentroid'], \
                                      amplitude = fit_init_par['peak'], \
                                      x_stddev= fit_init_par['fwhm']*n, \
                                      y_stddev= fit_init_par['fwhm']*n)
        fitter = fitting.LevMarLSQFitter()
        self._fit = fitter(fit_model, self._x, self._y, image)


    def fitSingleStarWithCentroid(self, ima):
        sy, sx= ima.shape
        y, x= np.mgrid[0:sy, 0:sx]
        cx=np.sum(ima*x)/np.sum(ima)
        cy=np.sum(ima*y)/np.sum(ima)
        self._fit= models.Gaussian2D(x_mean=(cx,), y_mean=(cy,),
                                     amplitude=(np.nan,), x_stddev=(np.nan,),
                                     y_stddev=(np.nan,))

        
    def getFitParameters(self):
        return self._fit
    
    
    def getCentroid(self):
        para= self.getFitParameters()
        return np.array([para.x_mean.value[0], para.y_mean.value[0]])
    
    
    def getSigmaXY(self):
        para= self.getFitParameters()
        return np.array([para.x_stddev.value[0], para.y_stddev.value[0]])


    def getAmplitude(self):
        para= self.getFitParameters()
        return para.amplitude.value[0]
    

    def getFittedImage(self):
        return self._fit(self._x, self._y)
    
