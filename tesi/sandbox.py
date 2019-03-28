'''
Created on 27 set 2018

@author: gcarla
'''

# import importlib
# importlib.reload(sandbox)
#from astropy.stats import sigma_clipped_stats
#from photutils.datasets import make_100gaussians_image
#from photutils import find_peaks
#from astropy.visualization import simple_norm


# import pyds9
# print(pyds9.ds9_targets())
# d = pyds9.DS9()
# d.set(" file /home/gcarla/reduction/luci1.20151218.0531.fits")


import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pickle
import ccdproc
from astropy.io import fits
from scipy.spatial import distance
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask
from astropy.table.table import Table
from photutils import Background2D, MedianBackground
from astropy.stats import SigmaClip
from photutils.detection.findstars import IRAFStarFinder
from astropy.stats import gaussian_fwhm_to_sigma
from astropy import units as u
from astropy.stats.funcs import gaussian_sigma_to_fwhm
from tesi import image_creator, image_fitter, data_reduction, image_aligner,\
    astrometricError_estimator, ePSF_builder, match_astropy_tables,\
    tablesOfFittedStars_creator,\
    differentialTJ_estimator, ngs_aligner
from ccdproc.ccddata import CCDData
from skimage.transform._warps import warp
from matplotlib.pyplot import xlabel, ylabel, xticks, yticks, plot, title
from cmath import pi
from scipy.ndimage.interpolation import shift
from astropy.nddata.utils import Cutout2D
from matplotlib.colors import LogNorm
from scipy.optimize.minpack import curve_fit


def getObjectNameFromFitsFile(filename):
    hdr = fits.open(filename)[0].header
    print(filename)
    try:
        return hdr['OBJECT']
    except KeyError:
        return 'na'


def getObjectPropertiesFromFitsFile(filename):
    try:
        print(filename)
        hdr = fits.open(filename)[0].header
    except:
        hdr={}
    try:
        obje= hdr['OBJECT']
    except KeyError:
        obje= 'na'
    try:
        ra= hdr['TELRA']
    except KeyError:
        ra= 'na'
    try:
        dec= hdr['TELDEC']
    except KeyError:
        dec= 'na'
    try:
        date= hdr['DATE']
    except KeyError:
        date= 'na'
    try:
        frametype= hdr['FRAMETYP']
    except KeyError:
        frametype= 'na'
    try:
        filter= hdr['FILTER']
    except KeyError:
        filter= 'na'

    return {'filename': filename, 'object': obje, 'ra': ra, 'dec': dec,
            'date': date, 'frametype': frametype, 'filter': filter}


def getFullNamesOfFitsFilesInSubFolder(topDir):

    fullnames= []
    exten= ".fits"
    for dname, names, files in os.walk(topDir):
        for name in files:
            if name.lower().endswith(exten):
                fullnames.append(os.path.join(dname, name))
    return fullnames

ARGOS_ARCHIVE_TOP_DIR= '/home/gcarla/tera1'


def getFullNamesOfAllLuciFitsFilesInSubFolder(topDir=ARGOS_ARCHIVE_TOP_DIR):

    fullnames= []
    for dname, names, files in os.walk(topDir):
        for name in files:
            if re.search('^luci[12]\.[0-9]{8}\.[0-9]{4}\.fits$',
                         os.path.basename(name.lower())):
                fullnames.append(os.path.join(dname, name))
    return fullnames


def getObjectsFromAllLuciFitsFiles(topDir=ARGOS_ARCHIVE_TOP_DIR):
    allObject= [getObjectPropertiesFromFitsFile(fname) for fname in
                getFullNamesOfAllLuciFitsFilesInSubFolder(topDir)]
    return allObject


def filterObjectDictionaryForProperty(objDict, propertyName, propertyValue):
    ''' quello che fa la funzione

    ngc2419Dict= filterObjectDictionaryForProperty(allObjectDict, 'object', 'NGC2419')

    '''
    filteredRes= []
    for meas in objDict:
        if meas[propertyName] == propertyValue:
            filteredRes.append(meas)
    return filteredRes


def saveObjectListToFile(objList, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(objList, handle, protocol=pickle.HIGHEST_PROTOCOL)


def restoreObjectListFromFile(filename):
    with open(filename, 'rb') as handle:
        objList= pickle.load(handle)
    return objList
    # data= sandbox.restoreObjectFromFile(...):
    # for file in data:
    #shutil.copy(file, directoryTargetPath)


def getObjectsFromListOfFitsFiles(listOfFitsFileNames):
    allObject= [getObjectPropertiesFromFitsFile(fname) for fname in
                listOfFitsFileNames]
    return allObject


def createRangeOfFileNames(radixName, fromIndex, toIndex):
    '''
    fileNames= createRangeOfFileNames(
        "/home/gcarla/tera1/201610/data/20161019/luci1.20161019", 759, 793)
    '''
    return ['%s.%04d.fits' % (radixName, idx) for idx in
            np.arange(fromIndex, toIndex+1)]


def createCubeFromFileNames(filenames):
    return np.dstack([fits.getdata(file) for file in filenames])


def main181008():
    radix= "/home/gcarla/tera1/201610/data/20161019/luci1.20161019"

    'DARKCUBE'
    darkFileNames= createRangeOfFileNames(radix, 4, 23)
    darkCube= createCubeFromFileNames(darkFileNames)
    _computeDarkImage= np.median(darkCube, axis=2)

    'SKYJCUBE'
    skyJFileNames= createRangeOfFileNames(radix, 759, 793)
    skyJCube= createCubeFromFileNames(skyJFileNames)
    skyJ= np.median(skyJCube, axis=2)

    'DARK SUBTRACTION FROM SKY'
    skyJ_dark = skyJ-_computeDarkImage

    'FLATJ FILE LIST'
    flatJFileNames= createRangeOfFileNames(radix, 75, 85)
    flatJFileList = []
    for filename in flatJFileNames:
        flatJFileList.append(fits.getdata(filename))

    'DARK SUBTRACTION FROM FLATJ LIST'
    flatJ_darkList = []
    for im in flatJFileList:
        flatJ_darkList.append(im-_computeDarkImage)

    'FLATJ CUBE AND (NORM)MASTERFLAT'
    flatJ_darkCube = np.dstack(file for file in flatJ_darkList)
    masterFlatJ = np.median(flatJ_darkCube, axis=2)
    normalizedMasterFlatJ = masterFlatJ/np.mean(masterFlatJ)

    'NGC2419(J) FILE LIST'
    scienceJFileNames_dith1= createRangeOfFileNames(radix, 732, 740)
    scienceJFileList = []
    for filename in scienceJFileNames_dith1:
        scienceJFileList.append(fits.getdata(filename))

    'DARK SUBTRACTION FROM NGC2419(J)'
    scienceJ_darkList = []
    for im in scienceJFileList:
        scienceJ_darkList.append(im-_computeDarkImage)

    'SKY SUBTRACTION FROM NGC2419(J)'
    skysub = []
    for im in scienceJ_darkList:
        skysub.append(im-skyJ_dark)

    'FINAL (MANCA IL FLAT!!!) NGC2419'
    scienceCube = np.dstack(im for im in skysub)
    scienceFinal = np.median(scienceCube, axis=2)

    return scienceFinal, _computeDarkImage, skyJ, normalizedMasterFlatJ, scienceJFileList
    # return _computeDarkImage, skyJ, scienceJ_darkList
    # return flatJ_darkCube, masterFlatJ, normalizedMasterFlatJ

#         , scienceJCube_dith2, \
#         scienceJ_dith2, scienceJ_dith3, scienceJCube_dith3


def cutSquareROI(ima, bl, sz):
    return ima[bl[0]:bl[0]+sz, bl[1]:bl[1]+sz]


def selectStars():

    ima_A = fits.getdata(
        '/home/gcarla/tera1/201512/December_reduced_iskren/NGC2419_H.fits')
    ima_B = fits.getdata(
        '/home/gcarla/tera1/201610/reduction/NGC2419_201610_19_23_24_26_H_reg.fits')

    sz=30
    blsA= np.array(
        [[1460, 890], [1295, 1315], [1415, 1960], [1960, 1735], [1750, 1765]])
    imaCutA = _extractSubIma(ima_A, blsA, sz)

    blsB= np.array(
        [[785, 495], [535, 875], [525, 1530], [1105, 1415], [900, 1405]])
    imaCutB = _extractSubIma(ima_B, blsB, sz)

    return imaCutA, imaCutB, blsA, blsB


#     ima_A_cut1 = ima_A ??????
#     ima_A_cut2 = ima_A[1295:1325,1315:1345]
#     ima_A_cut3 = ima_A[1415:1445,1960:1990]
#     ima_A_cut4 = ima_A[1960:1990,1735:1765]
#     ima_A_cut5 = ima_A[1750:1780,1765:1795]
#
#     ima_B_cut1 = ima_B[785:815,495:525]
#     ima_B_cut2 = ima_B[535:565,875:905]
#     ima_B_cut3 = ima_B[525:555,1530:1560]
#     ima_B_cut4 = ima_B[1105:1135,1415:1445]
#     ima_B_cut5 = ima_B[900:930,1405:1435]


def _extractSubIma(ima, bottomLeftCorners, sz):

    nImas= bottomLeftCorners.shape[0]
    imaCut= np.zeros((nImas, sz, sz))
    for i in np.arange(nImas):
        imaCut[i, :, :] = cutSquareROI(ima, bottomLeftCorners[i], sz)
    return imaCut


def showNorm(imaOrCcd, **kwargs):
    from astropy.visualization import imshow_norm, SqrtStretch
    from astropy.visualization.mpl_normalize import PercentileInterval
    from astropy.nddata import CCDData

    plt.clf()
    fig= plt.gcf()
    if isinstance(imaOrCcd, CCDData):
        arr= imaOrCcd.data
        wcs= imaOrCcd.wcs
        if wcs is None:
            ax= plt.subplot()
        else:
            ax= plt.subplot(projection=wcs)
            ax.coords.grid(True, color='white', ls='solid')
    else:
        arr= imaOrCcd
        ax= plt.subplot()
    if 'interval' not in kwargs:
        kwargs['interval']= PercentileInterval(99.7)
    if 'stretch' not in kwargs:
        kwargs['stretch']= SqrtStretch()
    if 'origin' not in kwargs:
        kwargs['origin']= 'lower'

    im, _= imshow_norm(arr, ax=ax, **kwargs)

    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize=11)


def showImaLuciInArcsec(ima, titleString=None, **kwargs):
    showNorm(ima, extent=[-120, 120, -120, 120], **kwargs)
    xlabel('arcsec', size=12)
    ylabel('arcsec', size=12)
    xticks(size=11)
    yticks(size=11)
    if titleString is not None:
        title(titleString)


def showImaLuciInPixels(ima, titleString=None, **kwargs):
    showNorm(ima, extent=[-1024, 1024, -1024, 1024], **kwargs)
    xlabel('pixel', size=12)
    ylabel('pixel', size=12)
    xticks(size=11)
    yticks(size=11)
    if titleString is not None:
        title(titleString)


def plotArrowsInLuciFOVInPixels(xPix, yPix, d_xPix, d_yPix, n=300):
    #    xPix = xPix-1024
    #    yPix = yPix-1024
    plt.plot(xPix, yPix, '.', color='r', markersize=0.1)
#    plt.xlim(-1024, 1024)
#    plt.ylim(-1024, 1024)
    plt.xlim(0, 2048)
    plt.ylim(0, 2048)
    plt.xlabel('px', size=13)
    plt.ylabel('px', size=13)
    plt.xticks(size=12)
    plt.yticks(size=12)
    for i in range(len(d_xPix)):
        plt.arrow(x=xPix[i], y=yPix[i], dx=n*d_xPix[i], dy=n*d_yPix[i],
                  head_width=20, head_length=20, color='r')


def plotDisplacementsInLUCI1sFoV_oneColor(xC, yC, d_x, d_y, color, n=300):
    xC_new = (xC-1024)*0.119
    yC_new = (yC-1024)*0.119
    dx_new = d_x*0.119
    dy_new = d_y*0.119
    plt.plot(xC_new, yC_new, '.', color=color, markersize=0.1)
    plt.xlim(-120, 120)
    plt.ylim(-120, 120)
    for i in range(len(d_x)):
        plt.arrow(x=xC_new[i], y=yC_new[i], dx=n*dx_new[i], dy=n*dy_new[i],
                  head_width=2, head_length=2, color=color)
    plt.xlabel('arcsec', size=13)
    plt.ylabel('arcsec', size=13)
    plt.xticks(size=12)
    plt.yticks(size=12)
    # LEGEND???


def plotDisplacementsInLUCI1sFoV_multiColor(xC, yC, d_x, d_y, w=0.007, n=1,
                                            colorb='yes', pathStr=None):
    xC_new = (xC-1024)*0.119
    yC_new = (yC-1024)*0.119
    if type(d_x) == list:
        dx_new = []
        dy_new = []
        for dx in d_x:
            dx_new.append(dx*0.119)
        for dy in d_y:
            dy_new.append(dy*0.119)
    else:
        dx_new = d_x*0.119
        dy_new = d_y*0.119
    colors = np.hypot(dx_new*1e03, dy_new*1e03)
    plt.quiver(xC_new, yC_new, dx_new*n, dy_new*n, colors, width=w)
    if colorb=='yes':
        cb = plt.colorbar()
        cb.set_label('$\Delta$ [mas]', rotation=90, size=10)
    plt.xlabel('arcsec', size=13)
    plt.ylabel('arcsec', size=13)
    plt.xticks(size=12)
    plt.yticks(size=12)
    if pathStr is not None:
        plt.savefig(pathStr)


def plotDisplacemensAndSave(xc, yc, dx_list, dy_list, color, pathStr=None):
    for i in range(len(dx_list)):
        plotDisplacements(xc, yc, dx_list[i], dy_list[i], color)
        plt.xlim(0, 2048)
        plt.ylim(0, 2048)
        plt.xlabel('px', size=13)
        plt.ylabel('px', size=13)
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.savefig(pathStr+'/%d' %(i))
        plt.close()


def plotArrowsColoredByIma(xMean, yMean, dx_list, dy_list):
    number = len(dx_list)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    for i in range(len(dx_list)):
        plotDisplacementsInLUCI1sFoV_oneColor(xMean,
                                              yMean,
                                              dx_list[i],
                                              dy_list[i],
                                              colors[i])


# def _showNormOld(ima, **kwargs):
#     from astropy.visualization import simple_norm
#     plt.clf()
#     norm= simple_norm(ima, 'linear', percent=99.5)
#     plt.imshow(ima, origin='lower', norm=norm)
#     plt.colorbar()


def tricksForBetterImasToShow(ccdIma):
    n = np.argwhere(ccdIma.mask == True)
    y = n[:, 0]
    x = n[:, 1]
    ccdIma.data[y, x] = np.median(ccdIma.data)

    def _cutForMedianCalculation(ima, x, y):
        cut = Cutout2D(ima, (x, y), (3, 3))
        return cut

    for i in range(y.shape[0]):
        ccdIma.data[y[i], x[i]] = np.median(_cutForMedianCalculation(
            ccdIma.data, y[i], x[i]).data)

    m = np.argwhere(ccdIma.data < 0)
    yy = m[:, 0]
    xx = m[:, 1]
    ccdIma.data[yy, xx] = np.median(ccdIma.data)
    return ccdIma


def measureStarsDistance(coord):
    distances= distance.cdist(coord, coord, 'euclidean')
    return distances


def main181022():
    expectedFwhm=2
    wantedSnr=10

    imaCutA, imaCutB, blsA, blsB= selectStars()
    imaCutAclean= np.array([removebkg(ima)[2] for ima in imaCutA])
    imaCutBclean= np.array([removebkg(ima)[2] for ima in imaCutB])

    fitA= [fitSingleStarWithGaussianFit(
        ima, wantedSnr*np.std(ima), expectedFwhm) for ima in imaCutAclean]
    fitB= [fitSingleStarWithGaussianFit(
        ima, wantedSnr*np.std(ima), expectedFwhm) for ima in imaCutBclean]

    coordA= np.squeeze(np.array(
        [[fitA[i][0].x_mean + blsA[i][1], fitA[i][0].y_mean + blsA[i][0]]
         for i in range(5)]))

    coordB= np.squeeze(np.array(
        [[fitB[i][0].x_mean + blsB[i][1], fitB[i][0].y_mean + blsB[i][0]]
         for i in range(5)]))

    return fitA, fitB, blsA, blsB, imaCutAclean, imaCutBclean, coordA, coordB


#     blA2= [1295, 1315]
#     blA2= [1295, 1315]
#     ima_A_cut1 = cutSquareROI(ima_A, blA1, sz)
#
#
#
#     ima_A_List = [ima_A_cut1, ima_A_cut2, ima_A_cut3, ima_A_cut4, ima_A_cut5]
#     ima_B_List = [ima_B_cut1, ima_B_cut2, ima_B_cut3, ima_B_cut4, ima_B_cut5]
#     ima_List = [ima_A_cut1, ima_A_cut2, ima_A_cut3, ima_A_cut4, ima_A_cut5, \
#                 ima_B_cut1, ima_B_cut2, ima_B_cut3, ima_B_cut4, ima_B_cut5]
#     #return ima_A_cut1, ima_A_cut2, ima_A_cut3, ima_A_cut4, ima_A_cut5
#     return ima_List, ima_A_List, ima_B_List


def removebkg(ima, snr=2, npixels=5, dilate_size=7):
    mask= make_source_mask(ima, snr, npixels, dilate_size)
    mean, median, std= sigma_clipped_stats(ima, sigma=3.0, mask=mask)

    return mean, median, std, ima-median


def removeBackground(ima, box_size, filter_size):

    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(ima, box_size=box_size, filter_size=filter_size,
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    return bkg  # , ima-bkg.background


# def fitSingleStarWithGaussianFit(ima_clean, threshold, fwhm):
#
#     dim_x, dim_y = ima_clean.shape
#     n = gaussian_fwhm_to_sigma
#     y, x = np.mgrid[:dim_y,:dim_x]
#
#     iraffind = IRAFStarFinder(threshold=threshold, fwhm=fwhm)
#     fit_init_par = iraffind.find_stars(ima_clean)
#
#     if len(fit_init_par) == 0:
#         raise Exception("No star found - (add info please)")
#     elif len(fit_init_par) > 1:
#         fit_init_par= fit_init_par[0]
#
#     fit_model = models.Gaussian2D(x_mean = fit_init_par['xcentroid'], \
#                                   y_mean = fit_init_par['ycentroid'], \
#                                   amplitude = fit_init_par['peak'], \
#                                   x_stddev= fit_init_par['fwhm']*n, \
#                                   y_stddev= fit_init_par['fwhm']*n)
#     fitter = fitting.LevMarLSQFitter()
#     fit = fitter(fit_model, x, y, ima_clean)
#
#     return fit, fit(x,y)
#

# def createGaussianImage():
#     table = Table()
#     shape= (30,30)
#     xMean= 11.56613378
#     yMean= 11.90298421
#     sx= 2.64317586
#     #sy= np.random.uniform(1,10)  'Asymmetric gaussian'
#     sy= sx
#     amp= 6000
#
#     table['x_mean']= [xMean]
#     table['y_mean']= [yMean]
#     table['x_stddev']= [sx]
#     table['y_stddev']= [sy]
#     table['amplitude']= [amp]
#     image1= make_gaussian_sources_image(shape, table)
#     image2= apply_poisson_noise(image1, random_state=12345)
#     image3= image2 + make_noise_image(shape, type= 'gaussian', mean= 0, stddev= 10)
#     return image1, image2, image3


# def createMultipleGaussianImage(nStars=100, shape=(2048,2048)):
#     table = Table()
#     xMean= np.random.uniform(1,shape[1]-1, nStars)
#     yMean= np.random.uniform(1,shape[0]-1, nStars)
#     sx= np.random.uniform(1, 1.5, nStars)
#     #sy= np.random.uniform(1,10)  'Asymmetric gaussian'
#     sy= sx
#     amp= np.random.uniform(5000,7000, nStars)
#
#     table['x_mean']= xMean
#     table['y_mean']= yMean
#     table['x_stddev']= sx
#     table['y_stddev']= sy
#     table['amplitude']= amp
#     image1= make_gaussian_sources_image(shape, table)
#     image2= image1 + make_noise_image(shape, type= 'gaussian', mean= 2000., stddev=9)
#     return image1, image2


def fitCrowdedFields(ima):
    iraffind= IRAFStarFinder(threshold=5000, fwhm=4, minsep_fwhm=0.01,
                             roundhi=5, roundlo=-5, sharplo=0.0, sharphi=2.0)
    fit_init= iraffind.find_stars(ima)
    #from photutils import CircularAperture
    #import matplotlib.pyplot as plt
    #apertures= CircularAperture(positions, r= 4.)
    # show(ima)
    #apertures.plot(color= 'blue')
    return fit_init

# clf(); imshow(np.sqrt(np.median(skyCube, axis=2)), origin='lower'); colorbar()


class FitAccuracyEstimator(object):

    def __init__(self, flux=2000, fwhm=2):
        self._shape= (42, 54)
        self._wantedFwhm= fwhm
        self._wantedFlux= flux

        self._fwhmEstimateForStarFinder= self._wantedFwhm
        self._thresholdInAmplitudeForStarFinder= \
            0.5 * self._guessAmplitudeFromFluxAndFwhm(self._wantedFlux,
                                                      self._wantedFwhm)
        self._imfit= ImageFitter(self._thresholdInAmplitudeForStarFinder,
                                 self._fwhmEstimateForStarFinder)
        self._imageCreator= ImageCreator(self._shape)

    def _guessAmplitudeFromFluxAndFwhm(self, flux, fwhm):
        return flux/(2*np.pi* (fwhm*gaussian_fwhm_to_sigma)**2)

    def onSingleGaussianWithPoissonNoise(self,
                                         usePoissonNoise=True,
                                         useCentroid=False):
        self._imageCreator.usePoissonNoise(usePoissonNoise)

        nIter= 100
        self.deltaPosX= np.zeros(nIter)
        self.deltaPosY= np.zeros(nIter)
        self.deltaSigmaX= np.zeros(nIter)
        self.deltaSigmaY= np.zeros(nIter)
        self.deltaFlux= np.zeros(nIter)

        for i in range(nIter):
            fwhm= self._wantedFwhm
            posX= np.random.uniform(fwhm, self._shape[1]-fwhm)
            posY= np.random.uniform(fwhm, self._shape[0]-fwhm)
            sigmaX= np.random.uniform(0.95*fwhm, 1.05*fwhm
                                      ) * gaussian_fwhm_to_sigma
            sigmaY= np.random.uniform(0.95*fwhm, 1.05*fwhm
                                      ) * gaussian_fwhm_to_sigma
            flux= np.random.uniform(
                0.9*self._wantedFlux,
                1.1*self._wantedFlux)
            ima= self._imageCreator.createGaussianImage(
                posX, posY, sigmaX, sigmaY, flux)

            try:
                if useCentroid:
                    self._imfit.fitSingleStarWithCentroid(ima)
                else:
                    self._imfit.fitSingleStarWithGaussianFit(ima)
                self.deltaPosX[i]= posX-self._imfit.getCentroid()[0]
                self.deltaPosY[i]= posY-self._imfit.getCentroid()[1]
                self.deltaSigmaX[i]= sigmaX-self._imfit.getSigmaXY()[0]
                self.deltaSigmaY[i]= sigmaY-self._imfit.getSigmaXY()[1]
                self.deltaFlux[i]= flux-self._imfit.getAmplitude()
            except Exception as e:
                print("got exception %s. Params: pos %g,%g "
                      "sigma %g,%g flux %g" %
                      (str(e), posX, posY, sigmaX, sigmaY, flux))
                self.deltaPosX[i]= np.nan
                self.deltaPosY[i]= np.nan
                self.deltaSigmaX[i]= np.nan
                self.deltaSigmaY[i]= np.nan
                self.deltaFlux[i]= np.nan

        # return ima


def centroid(ima):
    sy, sx= ima.shape
    y, x= np.mgrid[0:sy, 0:sx]
    cx=np.sum(ima*x)/np.sum(ima)
    cy=np.sum(ima*y)/np.sum(ima)
    return (cx, cy)


def fixHeadersKeywords():
    fileList=[]
    hdrList= []
    frametypList=[]

    for root, subdirs, files in os.walk('C:/Users/giulia/20161019/data'):
        for name in files:
            fileList.append(os.path.join(root, name))

#     for files in fileList:
#         hdr= fits.open(files)[0].header
#         hdrList.append(hdr)

#     for hdr in hdrList:
#         try:
#             frametyp= hdr['FRAMETYP']
#             frametypList.append(frametyp)
#         except KeyError:
#             print(hdr['FILENAME'])
#
#     for hdr in hdrList:
#         try:
#             frametyp= hdr['FRAMETYP']
#             frametypList.append(frametyp)
#         except KeyError:
#             frametyp= hdr['DATATYPE']
#             frametypList.append(frametyp)
#
#     for hdr in hdrList:
#         try:
#             frametyp= hdr['FRAMETYP']
#         except KeyError:
#             hdr.rename_keyword('DATATYPE', 'FRAMETYP')
#
#     for hdr in hdrList:
#         frametyp= hdr['FRAMETYP']
#         frametypList.append(frametyp)
#
    for files in fileList:
        hdr= fits.open(files)[0].header
        hdrList.append(hdr)
        for hdr in hdrList:
            try:
                frametyp= hdr['FRAMETYP']
            except KeyError:
                hdr.rename_keyword('DATATYPE', 'FRAMETYP')
            hdr.tofile(files, overwrite=True)

    return


def makeMask(ccddata_list, n):
    frames=np.array([d.data for d in ccddata_list])
    framesStd= frames.std(axis=0)
    # imshow(darksStd)
    #hist, hbin= np.histogram(darksStd, bins=100, range=(0, 30))
    #plot(hbin[1:], hist, '.-')
    loStd= np.median(framesStd) - n*framesStd.std()
    hiStd= np.median(framesStd) + n*framesStd.std()
    maskStd=np.zeros(framesStd.shape)
    maskStd[np.where(framesStd<loStd)]=1
    maskStd[np.where(framesStd>hiStd)]=1

    framesMedian= np.median(frames, axis=0)
    #hist, hbin= np.histogram(darksMedian, bins=100)
    #plot(hbin[1:], hist, '.-')
    loMedian= np.median(framesMedian) - n*framesMedian.std()
    hiMedian= np.median(framesMedian) + n*framesMedian.std()
    maskMedian=np.zeros(framesMedian.shape)
    maskMedian[np.where(framesMedian>hiMedian)]=1
    maskMedian[np.where(framesMedian<loMedian)]=1

    maskAll=np.clip(maskMedian+maskStd, 0, 1)
    frameMedianMasked= np.ma.masked_array(framesMedian, mask=maskAll)

    return frameMedianMasked, maskAll


def twoImagesAlignment():
    fname1 = '/home/gcarla/tera1/201610/data/20161019/luci1.20161019.0732.fits'
    fname2 = '/home/gcarla/tera1/201610/data/20161019/luci1.20161019.0741.fits'
    ima1_floatType = fits.getdata(fname1).astype(float)
    ima2_floatType = fits.getdata(fname2).astype(float)
    sz = 40
    bls = np.array([[1860, 1800], [380, 530], [560, 1780]])

    ima1_cut = _extractSubIma(ima1_floatType, bls, sz)
    ima2_cut = _extractSubIma(ima2_floatType, bls, sz)
    ima1_cutClean = np.array([removebkg(ima)[3] for ima in ima1_cut])
    ima2_cutClean = np.array([removebkg(ima)[3] for ima in ima2_cut])
    im_fit = ImageFitter(thresholdInPhotons=8000,
                         fwhm=4, min_separation=1,
                         sharplo=0.2, sharphi=1.0,
                         roundlo=-1.0, roundhi=1.0,
                         peakmax=5e04)

    pxcrd1 = []
    for im in ima1_cutClean:
        im_fit.fitSingleStarWithGaussianFit(im)
        pxcrd1.append(im_fit.getCentroid())
        pxcrd2 = []
    for im in ima2_cutClean:
        im_fit.fitSingleStarWithGaussianFit(im)
        pxcrd2.append(im_fit.getCentroid())

    x1 = np.array([pxcrd1[0][0], pxcrd1[1][0], pxcrd1[2][0]])
    x2 = np.array([pxcrd2[0][0], pxcrd2[1][0], pxcrd2[2][0]])
    y1 = np.array([pxcrd1[0][1], pxcrd1[1][1], pxcrd1[2][1]])
    y2 = np.array([pxcrd2[0][1], pxcrd2[1][1], pxcrd2[2][1]])
    shift_x = 2 - (x1.mean() - x2.mean())
    shift_y = abs(2 + y1.mean() - y2.mean())

    ima2_t = np.roll(ima2_floatType, (2, -2), axis=(1, 0))
    a_11 = ima2_t
    a_12 = np.roll(ima2_t, -1, axis=0)
    a_21 = np.roll(ima2_t, -1, axis=1)
    a_22 = np.roll(ima2_t, (-1, -1), axis=(1, 0))

    ima2Toima1 = a_12*(1-shift_x)*shift_y + a_22*shift_x*shift_y + \
        a_11*(1-shift_x)*(1-shift_y) + a_21*(shift_x)*(1-shift_y)

    return ima2Toima1, ima1_floatType, ima2_floatType


def main181228():
    drFileName= '/home/gcarla/DataReduction20161019.pkl'
    from tesi.data_reduction import DataReduction
    dr = DataReduction.restoreFromFile(drFileName)
    dr.setFilterType('J')
    dr.setIntegrationTime(3.0)
    dr.setObjectName('NGC2419')
    return dr

# TODO: Linearity and persistence analysis
# ADUlin = ADUraw + 4.155x10^(-6)(ADUraw)^2


def main181230(darks, flats, skys, sciences):
    from tesi.detector import LuciDetector
    from ccdproc import Combiner
#
#     dr= main181228()
#     darks= dr._darkIma
#     flats= dr._flatIma
#     skys= dr._skyIma
#     sciences= dr._scienceIma

#     drFileName = '/home/gcarla/tera1/201610/data/20161019'
#     dr = data_reduction.DataReduction(drFileName)
#     darks= dr.restoreFromFile(
#         '/home/gcarla/20161019_dataToRestore/darkImagesList.pkl')
#     flats= dr.restoreFromFile(
#         '/home/gcarla/20161019_dataToRestore/flatJ_ImagesList.pkl')
#     skys= dr.restoreFromFile(
#         '/home/gcarla/20161019_dataToRestore/skyJ_ImagesList.pkl')
#     sciences= dr.restoreFromFile(
#         '/home/gcarla/20161019_dataToRestore/scienceJ_ImagesList.pkl')

    # deviation, serve a quacosa?
    def _computeDeviation(ccd0, detector):
        cnew= ccdproc.create_deviation(
            ccd0,
            gain=detector.gainAdu2Electrons*u.electron/u.adu,
            readnoise=detector.ronInElectrons*u.electron)
        return cnew
#    sciences0_new= _computeDeviation(sciences[0], LuciDetector())

    def _makeClippedCombiner(ccdData_list):
        ccdComb = Combiner(ccdData_list)
        medCcd = ccdComb.median_combine()
        minclip = np.median(medCcd) - 3*np.std(medCcd)
        maxclip = np.median(medCcd) + 3*np.std(medCcd)
        ccdComb.minmax_clipping(min_clip=minclip, max_clip=maxclip)
        return ccdComb

    def _makeMasterDark(darks):
        #         darkCombiner= Combiner(darks)
        #         darkCombiner.sigma_clipping(low_thresh=3, high_thresh=3,
        # func=np.ma.median, dev_func=np.ma.std)
        darkCombiner = _makeClippedCombiner(darks)
        masterDark= darkCombiner.median_combine()
        masterDark.header['exptime']= darkCombiner.ccd_list[
            0].header['exptime']
        masterDark.header['DIT']= darkCombiner.ccd_list[0].header['DIT']
        # TODO: something else to be added to the masterDark.header?
        return masterDark

    def _adu2Electron(ccd):
        # TODO: LUCI has 'gain' and 'rdnoise' keywords in the header. Use them
        # instead of relying on a LuciDetector object? (RON could change
        # in case of updated electronics?) Anyhow we decided to trust the
        # FITS header
        return ccdproc.gain_correct(ccd,
                                    LuciDetector().gainAdu2Electrons,
                                    u.electron/u.adu)

    def _trim(ccd):
        trimFitsSection= "DATASEC"
        return ccdproc.trim_image(ccd, ccd.header[trimFitsSection])

    def _makeMasterFlat(flats, masterDark):
        flatsDarkSubtracted=[]
        for flat in flats:
            flat= ccdproc.subtract_dark(
                flat, masterDark, exposure_time='DIT',
                exposure_unit=u.second,
                add_keyword={'HIERARCH GIULIA DARK SUB': True})
            flatsDarkSubtracted.append(flat)

        flatCombiner= _makeClippedCombiner(flatsDarkSubtracted)
#         flatCombiner= Combiner(flatsDarkSubtracted)
#         flatCombiner.sigma_clipping(low_thresh=3, high_thresh=3,
#                                     func=np.ma.median, dev_func=np.ma.std)

        def scalingFunc(arr):
            return 1./np.ma.average(arr)

        flatCombiner.scaling= scalingFunc
        masterFlat= flatCombiner.median_combine()
        masterFlat.header= flatsDarkSubtracted[0].meta
        return masterFlat

    masterDark= _makeMasterDark(darks)
    masterFlat= _makeMasterFlat(flats, masterDark)
    masterFlatElectron= _adu2Electron(masterFlat)

    def _calibrateAndCombine(ccds, masterDark, masterFlatElectron):
        objCalibrated=[]
        for ccd in ccds:
            ccdDark= ccdproc.subtract_dark(ccd, masterDark,
                                           exposure_time='DIT',
                                           exposure_unit=u.second,
                                           add_keyword={'calib': 'subtracted dark'})
            ccdGain= _adu2Electron(ccdDark)
            ccdFlat= ccdproc.flat_correct(ccdGain, masterFlatElectron)
            objCalibrated.append(ccdFlat)

        objCombiner= Combiner(objCalibrated)
        medianObj= objCombiner.median_combine()
        return medianObj

    sky= _calibrateAndCombine(skys, masterDark, masterFlatElectron)
    sci= _calibrateAndCombine(sciences, masterDark, masterFlatElectron)
    scisky=sci.subtract(sky)

    def iterativeSigmaClipping(combiner):
        old= 0
        new= combiner.data_arr.mask.sum()
        print("new %d" % new)
        while(new>old):
            combiner.sigma_clipping(func=np.ma.median)
            old= new
            new= combiner.data_arr.mask.sum()
            print("new %d" % new)

    return scisky


def wcsTest(dr):
    """
    HIERARCH LBTO LUCI WCS CTYPE1 = 'RA---TAN' / the coordinate type and projection 
    HIERARCH LBTO LUCI WCS CTYPE2 = 'DEC--TAN' / the coordinate type and projection 
    HIERARCH LBTO LUCI WCS CRPIX1 = 1024. / the pixel coordinates of the reference p
    HIERARCH LBTO LUCI WCS CRPIX2 = 1024. / the pixel coordinates of the reference p
    HIERARCH LBTO LUCI WCS CRVAL1 = 205.7395 / the WCS coordinates on the reference 
    HIERARCH LBTO LUCI WCS CRVAL2 = 32. / the WCS coordinates on the reference point
    HIERARCH LBTO LUCI WCS CD1_1 = -3.3E-05 / the rotation matrix for scaling and ro
    HIERARCH LBTO LUCI WCS CD1_2 = 0. / the rotation matrix for scaling and rotation
    HIERARCH LBTO LUCI WCS CD2_1 = 0. / the rotation matrix for scaling and rotation
    HIERARCH LBTO LUCI WCS CD2_2 = 3.3E-05 / the rotation matrix for scaling and rot

    TELRA   = '07 38 9.002'        / Telescope Right Accention
    TELDEC  = '+38 53 11.504'      / Telescope Declination
    OBJRA   = '07 38 9.002'        / Target Right Ascension from preset
    OBJDEC  = '+38 53 11.504'      / Target Declination from preset
    OBJRADEC= 'FK5     '           / Target Coordinate System
    OBJEQUIN= 'J2000   '           / Target Coordinate System Equinox
    OBJPMRA =                   0. / Target RA proper motion [mas per yr]
    OBJPMDEC=                   0. / Target DEC proper motion [mas per yr]
    OBJEPOCH=                2000. / Target Epoch
    GUIRA   = '07 38 19.187'       / Guide Star RA
    GUIDEC  = '+38 55 15.648'      / Guide Star DEC
    AONAME  = 'N1288-0180843'      / AO Star Name
    AORA    = '07 38 15.702'       / AO Star RA
    AODEC   = '+38 53 33.108'      / AO Star DEC


    """
    from astropy import wcs
    from astropy.coordinates import SkyCoord, FK5
    sciences= dr._scienceIma
    ccd0=sciences[0]
    hdr0= ccd0.header
    ccd0wcs= wcs.WCS(hdr0)
    pxs= np.array([[0, 0], [1024, 1024], [512, 1024]], np.float)
    ccd0wcs.all_pix2world(pxs, 1)

    px= np.arange(ccd0.shape[1])
    py= np.arange(ccd0.shape[0])
    wx, wy= ccd0wcs.all_pix2world(px, py, 1)

    if hdr0['OBJRADEC'] == 'FK5':
        frameType= FK5()
    c=SkyCoord(ccd0.header['OBJRA'], ccd0.header['OBJDEC'],
               frame=frameType, unit=(u.hourangle, u.deg))

    # AO guide star. Find it in image
    aoStarCoordW=SkyCoord(ccd0.header['AORA'], ccd0.header['AODEC'],
                          frame=frameType, unit=(u.hourangle, u.deg))
    aoStarCoordPx= ccd0wcs.world_to_pixel(aoStarCoordW)


def calibrateScienceImageHavingMasterFrames(sci,
                                            masterdark,
                                            masterflat,
                                            mastersky):
    sci_dark = ccdproc.subtract_dark(
        sci,
        masterdark,
        exposure_time='DIT',
        exposure_unit=u.second,
        add_keyword={'HIERARCH GIULIA DARK SUB': True})
    sciFlat = ccdproc.flat_correct(sci_dark, masterflat)
    sciFinalInADU = sciFlat.subtract(mastersky)
    sciFinalInADU.header = sci.header

    def _adu2Electron(ccd):
        return ccdproc.gain_correct(ccd, ccd.header['GAIN'], u.electron/u.adu)

    return _adu2Electron(sciFinalInADU)

# def main190121(image):
#
#     #     genDet= GenericDetector((32, 32))
#     #     genDet.ronInElectrons=2.0
#     #     ima_cre=image_creator.ImageCreator((32, 32))
#     #     ima_cre.setDetector(genDet)
#     #
#     #     image= ima_cre.createGaussianImage(16, 16, 3.0, 3.0, 80000) + \
#     #         ima_cre.createGaussianImage(20, 16.5, 3.0, 3.0, 8000) + \
#     #         ima_cre.createGaussianImage(5,  20,  3.0, 3.0, 8000) + \
#     #         ima_cre.createGaussianImage(14, 8.5, 3.0, 3.0, 8000)
#
#     n = gaussian_sigma_to_fwhm
#     guessedSigma=2.0
#     highThresholdToGetBrightStarOnly= 100
#     psf_fitter = image_fitter.PSFphotometry(
#         highThresholdToGetBrightStarOnly,
#         guessedSigma*n, 1, 0.2, 1.0, -1.0, 1.0, (27, 27))
#
#     psf_fitter._psf_model.sigma.fixed=False
#     psf_fitter.setImage(image)
#     tab= psf_fitter.basicPSFphotometry()
#     bestSigma= float(np.median(tab['sigma_fit']))
#
#     psf_fitter = image_fitter.PSFphotometry(
#         None,
#         bestSigma*n, 1, 0.2, 1.0, -1.0, 1.0, (27, 27))
#
#     # psf_fitter._psf_model.sigma.fixed=False
#     psf_fitter.setImage(image)
#     tabAll= psf_fitter.basicPSFphotometry()
#     res= psf_fitter.basic_photom.get_residual_image()
#     return tabAll, res  # , image
#
#
# class main190125():
#
#     def __init__(self,
#                  image,
#                  thresholdForBrightStarsOnly,
#                  guessedSigma,
#                  minSeparationForUncrowdedStars):
#
#         self._image = image
#         self._initThreshold = thresholdForBrightStarsOnly
#         self._initSigma = guessedSigma
#         self._initMinSeparation = minSeparationForUncrowdedStars
#         self._n = gaussian_sigma_to_fwhm
#
#     def _computeBestSigma(self):
#
#         psf_fitter = image_fitter.PSFphotometry(
#             self._initThreshold,
#             self._initSigma*self._n, self._initMinSeparation,
#             0.2, 1.0, -1.0, 1.0, (45, 45), 50)
#
#         psf_fitter._psf_model.sigma.fixed=False
#         psf_fitter.setImage(self._image)
#         self._initTab= psf_fitter.basicPSFphotometry()
#         self._bestSigma= float(np.median(self._initTab['sigma_fit']))
#         return self._bestSigma
#
#     def doPhotometry(self):
#
#         psf_fitter = image_fitter.PSFphotometry(
#             None,
#             self._computeBestSigma()*self._n,
#             1, 0.2, 1.0, -1.0, 1.0, (45, 45), 50)
#
#         # psf_fitter._psf_model.sigma.fixed=False
#         psf_fitter.setImage(self._image)
#         tabAll= psf_fitter.basicPSFphotometry()
#         res= psf_fitter.basic_photom.get_residual_image()
#         return tabAll, res


# def main190130(ccdData_list):
#
#     #     darks = restoreObjectListFromFile(
#     #         '/home/gcarla/workspace/20161019_dataToRestore/darkImagesList.pkl')
#     ccdComb = Combiner(ccdData_list)
#     medCcd = ccdComb.median_combine()
# #     medMedDarks = np.median(medDarks)
# #     stdMedDarks = np.std(medDarks)
# #     minclip = medMedDarks - 3*stdMedDarks
# #     maxclip = medMedDarks + 3*stdMedDarks
#
#     minclip = np.median(medCcd) - 3*np.std(medCcd)
#     maxclip = np.median(medCcd) + 3*np.std(medCcd)
#
#     ccdComb.minmax_clipping(min_clip=minclip, max_clip=maxclip)
#     #ccdFinal = ccdComb.median_combine()
#
#     return ccdComb


# Ex. sort tables: sourceTabIma1.sort(['xcentroid'])

# def findTransformationMatrix(sourceTabIma1, sourceTabIma2):
#     PosX1 = np.array(sourceTabIma1['xcentroid'])
#     PosY1 = np.array(sourceTabIma1['ycentroid'])
#     PosX2 = np.array(sourceTabIma2['xcentroid'])
#     PosY2 = np.array(sourceTabIma2['ycentroid'])
#
#     PosMatr1 = np.array([PosX1, PosY1, np.ones(len(sourceTabIma1))])
#     PosMatr2 = np.array([PosX2, PosY2, np.ones(len(sourceTabIma2))])
#
#     TransfMat = np.dot(PosMatr2, np.linalg.pinv(PosMatr1))
#     # LeastSqTransfMat = np.linalg.lstsq(np.linalg.pinv(PosMatr2),
#     #                                   np.linalg.pinv(PosMatr1))
#
#     return TransfMat, PosMatr1, PosMatr2
#
#
# def alignImage(ccdImaToAlign, transfMatrix):
#     alignedIma = warp(ccdImaToAlign, transfMatrix,
#                       output_shape=ccdImaToAlign.shape,
#                       order=3, mode='constant',
#                       cval=np.median(ccdImaToAlign.data))
#     return alignedIma
    # TODO: alignedIma is numpy.ndarray. Return as CCDData?

class main190208():

    def __init__(self,
                 fitsname1,
                 fitsname2):

        self.ima1 = CCDData.read(fitsname1)
        self.ima2 = CCDData.read(fitsname2)

    def _getIma2AlignedtoIma1(self):
        im_align = image_aligner.ImageAligner(self.ima1, self.ima2)
        self.ima2Aligned = im_align.applyTransformation()
        return self.ima2Aligned

    def getStarsCoords(self):

        self._basic1 = fitWithEPSFModel(self.ima1)
        self._basic2 = fitWithEPSFModel(self.ima2)

        self.basic1Sort, self.basic2Sort = match2TablesPhotometry(
            self._basic1, self._basic2, 5)
        self.coords1 = np.vstack([np.array(self.basic1Sort['x_fit']),
                                  np.array(self.basic1Sort['y_fit'])]).T
        self.coords2 = np.vstack([np.array(self.basic2Sort['x_fit']),
                                  np.array(self.basic2Sort['y_fit'])]).T
        return self.coords1, self.coords2

    def measureDistances(self):
        distances = np.linalg.norm(
            self.getStarsCoords()[1]-self.getStarsCoords()[0],
            axis=1)
        plt.figure()
        plt.plot(distances, '.-')
        return distances


def main190213():
    fname = '/home/gcarla/workspace/20161019/reduction/FilterJ/'
    'IndividualFrames_Dither1/NGC2419_20161019_luci1_List.pkl'
    imasList = restoreObjectListFromFile(fname)
    est = astrometricError_estimator.EstimateAstrometricError(imasList)
    est.fitStarsOnAllFrames()
    est.createCubeOfStarsInfo()
    meanx = est.getMeanPositionX()
    meany = est.getMeanPositionY()
    stdx = est.getStdXinArcsecs()
    stdy = est.getStdYinArcsecs()
    astromError = np.sqrt(stdx**2 + stdy**2)

    area = 400*astromError
    colors = astromError
    plt.scatter(meanx, meany, s=area, c=colors)


def infToZero(ima):
    i = np.argwhere(np.isinf(ima))
    y = i[:, 0]
    x = i[:, 1]
    ima[y, x]=0
    return ima


def nanToZero(ima):
    i = np.argwhere(np.isnan(ima))
    y = i[:, 0]
    x = i[:, 1]
    ima[y, x]=0
    return ima


def main190216():
    imas = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/reduction/FilterJ/'
        'IndividualFrames_Dither1/NGC2419_20161019_luci1_ListFITS.pkl')
    imasList = []
    for ima in imas:
        im = infToZero(ima)
        im = nanToZero(ima)
        imasList.append(im)
    epsfImas = []
    fitTabs = []

    def _getEPSFOnSingleFrame(ima):
        builder = ePSF_builder.ePSFBuilder(threshold=1e04,
                                           fwhm=3.)
        builder.buildEPSF(ima)
        epsfModel = builder.getEPSFModel()
        return epsfModel

    def _fitStarsWithEPSFOnSingleFrame(ima,
                                       epsfModel,
                                       fitshape,
                                       apertureRadius):
        ima_fit = image_fitter.ImageFitter(thresholdInPhotons=1e03,
                                           fwhm=3., min_separation=3.,
                                           sharplo=0.1, sharphi=2.0,
                                           roundlo=-1.0, roundhi=1.0,
                                           peakmax=5e04)
        ima_fit.fitStarsWithBasicPhotometry(image=ima,
                                            model=epsfModel,
                                            fitshape=fitshape,
                                            apertureRadius=apertureRadius)
        fitTab = ima_fit.getFitTable()
        return fitTab

    for ima in imasList:
        tab = _fitStarsWithEPSFOnSingleFrame(ima, _getEPSFOnSingleFrame(ima),
                                             (55, 55), 55)
        fitTabs.append(tab)
        epsfImas.append(_getEPSFOnSingleFrame(ima).data)

    def _matchTablesList(tabsList):
        match = match_astropy_tables.MatchTables(2)
        refTab = match.match2TablesPhotometry(tabsList[0], tabsList[1])[0]
        for fitTab in tabsList[2:]:
            _, refTab = match.match2TablesPhotometry(fitTab, refTab)

        matchingFitTabs = []
        for fitTab in tabsList:
            tab1, _ = match.match2TablesPhotometry(fitTab, refTab)
            matchingFitTabs.append(tab1)
        return matchingFitTabs

    estimator = astrometricError_estimator.EstimateAstrometricError(
        _matchTablesList(fitTabs))
    estimator.createCubeOfStarsInfo()
    estimator.plotStandardAstroErrorOntheField(500)
    d0_x, d0_y = estimator.getDisplacementsFromMeanPositions(0)
    d1_x, d1_y = estimator.getDisplacementsFromMeanPositions(1)
    d2_x, d2_y = estimator.getDisplacementsFromMeanPositions(2)
    d3_x, d3_y = estimator.getDisplacementsFromMeanPositions(3)
    d4_x, d4_y = estimator.getDisplacementsFromMeanPositions(4)
    d5_x, d5_y = estimator.getDisplacementsFromMeanPositions(5)
    d6_x, d6_y = estimator.getDisplacementsFromMeanPositions(6)
    d7_x, d7_y = estimator.getDisplacementsFromMeanPositions(7)
    d8_x, d8_y = estimator.getDisplacementsFromMeanPositions(8)

    def plotDisp(dx, dy):
        estimator.plotDisplacements(dx, dy)
        plt.xlabel('px', size=13)
        plt.ylabel('px', size=13)
        plt.xticks(size=12)
        plt.yticks(size=12)

    def plotDispMinusTT(dx, dy):
        estimator.plotDisplacementsMinusTT(dx, dy)
        plt.xlabel('px', size=13)
        plt.ylabel('px', size=13)
        plt.xticks(size=12)
        plt.yticks(size=12)


def main190217():
    tabs = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/plot/main190216/matchingTablesList.pkl')

    posIma0 = np.vstack((np.array([np.array(tabs[0]['x_fit'])]),
                         np.array([np.array(tabs[0]['y_fit'])]))).T
    xRef = posIma0[:, 0]
    yRef = posIma0[:, 1]

    def getDisplacementFromFirstIma(i):
        posIma = np.vstack((np.array([np.array(tabs[i]['x_fit'])]),
                            np.array([np.array(tabs[i]['y_fit'])]))).T
        d = posIma-posIma0
        d_x = d[:, 0]
        d_y = d[:, 1]
        return d_x, d_y

    d1_x, d1_y = getDisplacementFromFirstIma(1)
    d2_x, d2_y = getDisplacementFromFirstIma(2)
    d3_x, d3_y = getDisplacementFromFirstIma(3)
    d4_x, d4_y = getDisplacementFromFirstIma(4)
    d5_x, d5_y = getDisplacementFromFirstIma(5)
    d6_x, d6_y = getDisplacementFromFirstIma(6)
    d7_x, d7_y = getDisplacementFromFirstIma(7)
    d8_x, d8_y = getDisplacementFromFirstIma(8)

    def plotDisplacements(d_x, d_y):
        plt.plot(xRef, yRef, '.', color='r', markersize=0.1)
        plt.xlim(0, 2048)
        plt.ylim(0, 2048)
        plt.xlabel('px', size=13)
        plt.ylabel('px', size=13)
        plt.xticks(size=12)
        plt.yticks(size=12)
        for i in range(len(d_x)):
            plt.arrow(x=xRef[i], y=yRef[i], dx=300*d_x[i], dy=300*d_y[i],
                      head_width=20, head_length=20, color='r')

    def plotDisplacementsMinusTT(d_x, d_y):
        plt.plot(xRef, yRef, '.', color='r', markersize=0.1)
        plt.xlim(0, 2048)
        plt.ylim(0, 2048)
        plt.xlabel('px', size=13)
        plt.ylabel('px', size=13)
        plt.xticks(size=12)
        plt.yticks(size=12)
        for i in range(len(d_x)):
            plt.arrow(x=xRef[i], y=yRef[i], dx=300*(d_x[i]-d_x.mean()),
                      dy=300*(d_y[i]-d_y.mean()), head_width=20,
                      head_length=20, color='r')


def main190218():
    tabs = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/plot/main190216/'
        'matchingTablesList.pkl')
    imas = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/reduction/FilterJ/'
        'IndividualFrames_Dither1/NGC2419_20161019_luci1_ListFITS.pkl')
    imasList = []
    for ima in imas:
        im = infToZero(ima)
        im = nanToZero(ima)
        imasList.append(im)

    def findTransfMatrixWithIma0(tab):
        ima_al = image_aligner.ImageAligner(tabs[0], tab)
        ima_al.findTransformationMatrixWithDAOPHOTTable()
        matr = ima_al.getTransformationMatrix()
        return matr

    m1 = findTransfMatrixWithIma0(tabs[1])
    m2 = findTransfMatrixWithIma0(tabs[2])
    m3 = findTransfMatrixWithIma0(tabs[3])
    m4 = findTransfMatrixWithIma0(tabs[4])
    m5 = findTransfMatrixWithIma0(tabs[5])
    m6 = findTransfMatrixWithIma0(tabs[6])
    m7 = findTransfMatrixWithIma0(tabs[7])
    m8 = findTransfMatrixWithIma0(tabs[8])

    def applyTransformation(ima, matrix):
        alignedIma = warp(ima, matrix,
                          output_shape=ima.shape,
                          order=3, mode='constant',
                          cval=np.median(ima.data))
        return alignedIma

    ima1_new = applyTransformation(imasList[1], m1)
    ima2_new = applyTransformation(imasList[2], m2)
    ima3_new = applyTransformation(imasList[3], m3)
    ima4_new = applyTransformation(imasList[4], m4)
    ima5_new = applyTransformation(imasList[5], m5)
    ima6_new = applyTransformation(imasList[6], m6)
    ima7_new = applyTransformation(imasList[7], m7)
    ima8_new = applyTransformation(imasList[8], m8)
    imasList_new = [ima1_new, ima2_new, ima3_new, ima4_new, ima5_new,
                    ima6_new, ima7_new, ima8_new]

    alignedImaTabs = []
    epsfAlignedImas = []
    for ima in imasList_new:
        tab = main190216()._fitStarsWithEPSFOnSingleFrame(
            ima, main190216()._getEPSFOnSingleFrame(ima), (55, 55), 55)
        alignedImaTabs.append(tab)
        epsfAlignedImas.append(main190216()._getEPSFOnSingleFrame(ima).data)
    tabs_new = [tabs[0], alignedImaTabs[0], alignedImaTabs[1],
                alignedImaTabs[2], alignedImaTabs[3], alignedImaTabs[4],
                alignedImaTabs[5], alignedImaTabs[6], alignedImaTabs[7]]
    matchTabs = main190216()._matchTablesList(tabs_new)


def fillBadValues(ima):

    def infToMean(ima):
        i = np.argwhere(np.isinf(ima))
        y = i[:, 0]
        x = i[:, 1]
        ima[y, x]=ima.mean()
        return ima

    def nanToMean(ima):
        i = np.argwhere(np.isnan(ima))
        y = i[:, 0]
        x = i[:, 1]
        ima[y, x]=ima.mean()
        return ima

    im = infToMean(ima)
    im = nanToMean(ima)
    return im


def main190220():

    def showSingleGaussImaWithPhotNoise(shape, posX=17.02, posY=20.5,
                                        flux=5e03, stdx=1.2, stdy=None):
        ima_cre = image_creator.ImageCreator(shape)
        ima_cre.usePoissonNoise(True)
        ima = ima_cre.createGaussianImage(posX, posY, flux, stdx)
        showNorm(ima)
        plt.xlabel('px', size=12)
        plt.ylabel('px', size=12)
        plt.xticks(size=11)
        plt.yticks(size=11)

    def estimateTheoreticalAstromError():
        test = theoretical_astrometricError_test.testTheoreticalAstrometricError()
        test.estimateAstromErrorForDifferentFluxes()
        err = test.getAstrometricErrorInPixels()
        flux = np.arange(1e03, 1e05, 1e03)
        ff = 1/np.sqrt(flux)
        fwhm = 1.2*gaussian_sigma_to_fwhm
        errxTheor = (fwhm*ff)/pi
        errAllTheor = np.sqrt(2*errxTheor**2)
        (err-errAllTheor).std()

        plot(flux, err, '.-', label="Fit")
        plt.legend()
        plot(flux, errAllTheor, '.-', label="Theory")
        plt.legend()
        plt.xticks(size=11)
        plt.yticks(size=11)
        plt.xlabel('Flux', size=12)
        plt.ylabel('Standard astrometric error [px]', size=12)


def buildEPSFsAndFitImaCuts(image):
    listOfImaCuts = [image[0:1024, 0:1024], image[1024:2048, 0:1024],
                     image[0:1024, 1024:2048], image[1024:2048, 1024:2048]]

    def _buildEPSF(ima):
        epsf = ePSF_builder.ePSFBuilder(ima, 5e03, 3.)
        epsf.removeBackground()
        epsf.extractStars()
        epsf.selectGoodStars()
        epsf.buildEPSF()
        epsfModel = epsf.getEPSFModel()
        return epsfModel

    epsfsList = []
    for ima in listOfImaCuts:
        epsfsList.append(_buildEPSF(ima))

    def _fitStars(ima, model):
        ima_fit = image_fitter.ImageFitter(1e03, 3., 3., 0.1, 2.0, -1.0, 1.0,
                                           5e04)
        ima_fit.fitStarsWithBasicPhotometry(ima, model, (21, 21), 45)
        fitTab = ima_fit.getFitTable()
        return fitTab

    tabsList = []
    for i in range(len(listOfImaCuts)):
        tabImaCut = _fitStars(listOfImaCuts[i], epsfsList[i])
        tabsList.append(tabImaCut)

    tabsList[1]['y_fit'] = tabsList[1]['y_fit'] + 1024
    tabsList[2]['x_fit'] = tabsList[2]['x_fit'] + 1024
    tabsList[3]['x_fit'] = tabsList[3]['x_fit'] + 1024
    tabsList[3]['y_fit'] = tabsList[3]['y_fit'] + 1024

    tabAllStars = tabsList[0].copy()
    tabAllStars.remove_rows(range(len(tabAllStars)))

    for tab in tabsList:
        for i in range(len(tab)):
            tabAllStars.add_row(tab[i])

    #del listOftabsImaCuts[0]
#     for tab in tabsList:
#         tabAllStars['x_fit'] = tabsList[i]['x_fit']
#         tabAllStars['y_fit'] = tabsList[i]['y_fit']

    return listOfImaCuts, epsfsList, tabAllStars


class IRAFStarFinderExcludingMaskedPixel(IRAFStarFinder):

    def __init__(self, *args, **kwargs):
        super(IRAFStarFinderExcludingMaskedPixel, self).__init__(
            *args, **kwargs)

    def _cutDataOnABoxAroundStar(self, ima, xc, yc):
        cut = Cutout2D(ima, (xc, yc), 21)
        return cut.data

    def find_stars(self, data, mask=None):
        self.table= super(IRAFStarFinderExcludingMaskedPixel, self).find_stars(
            data, mask)

        if data.mask.any():
            self.tableCopy= self.table.copy()
            self.tableCopy.remove_rows(range(len(self.tableCopy)))

            for i in range(len(self.table)-1):
                imaCut = self._cutDataOnABoxAroundStar(
                    data,
                    self.table[i]['xcentroid'],
                    self.table[i]['ycentroid'])
                if imaCut.mask.any()==False:
                    self.tableCopy.add_row(self.table[i])
            return self.tableCopy
        else:
            return self.table


def convertFittedPositionsInTabsListToArray(tabsList):
    '''
    TabsList: List Of N tables
               with positions (x_fit, y_fit) of M stars
    Returns: ndarray with shape (N, M, 2)
    '''
    return np.dstack((
        np.array([np.array(tab['x_fit']) for tab in tabsList]),
        np.array([np.array(tab['y_fit']) for tab in tabsList])))


def maskCrowdedRegion(imaList):
    for ima in imaList:
        ima.mask[600:1300, 700:1400] = True
    return imaList


def main190311():

    def tiltJitter_Dither1_FilterJ():
        tabs = restoreObjectListFromFile('/home/gcarla/workspace/20161019/'
                                         'DataToRestore_forAnalysis/FilterJ/'
                                         'Dither1/matchingTabs_list.pkl')
        diff = differentialTJ_estimator.estimateDifferentialTJ(tabs, [414, 0])
        ae= diff.astrometricError()
        plt.figure()
        plot(diff.polCoord[0]*0.1178, ae[:, 0]*0.1178*1e03,
             '.', label="$\sigma_{\parallel}$")
        plt.legend()
        plot(diff.polCoord[0]*0.1178, ae[:, 1]*0.1178*1e03,
             '.', label="$\sigma_{\perp}$")
        plt.legend()
        plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
        plt.ylabel('$\sigma_{tilt\:jitter}$ [mas]', size=12)
        plt.xticks(size=11)
        plt.yticks(size=11)
        plt.title('Filtro J')

    def tiltJitter_Dither1_FilterH():
        tabs = restoreObjectListFromFile('/home/gcarla/workspace/20161019/'
                                         'DataToRestore_forAnalysis/FilterH/'
                                         'Dither1/matchingTabs_list.pkl')
        diff = differentialTJ_estimator.estimateDifferentialTJ(tabs, [414, 0])
        ae= diff.astrometricError()
        plt.figure()
        plot(diff.polCoord[0]*0.1178, ae[:, 0]*0.1178*1e03,
             '.', label="$\sigma_{\parallel}$")
        plt.legend()
        plot(diff.polCoord[0]*0.1178, ae[:, 1]*0.1178*1e03,
             '.', label="$\sigma_{\perp}$")
        plt.legend()
        plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
        plt.ylabel('$\sigma_{tilt\:jitter}$ [mas]', size=12)
        plt.xticks(size=11)
        plt.yticks(size=11)
        plt.ylim(0, 39)
        plt.title('Filtro H')

    def tiltJitter_Dither1_FilterKs():
        tabs = restoreObjectListFromFile('/home/gcarla/workspace/20161019/'
                                         'DataToRestore_forAnalysis/FilterKs/'
                                         'Dither1/matchingTabs_list.pkl')
        diff = differentialTJ_estimator.estimateDifferentialTJ(tabs, [414, 0])
        ae= diff.astrometricError()
        plt.figure()
        plot(diff.polCoord[0]*0.1178, ae[:, 0]*0.1178*1e03,
             '.', label="$\sigma_{\parallel}$")
        plt.legend()
        plot(diff.polCoord[0]*0.1178, ae[:, 1]*0.1178*1e03,
             '.', label="$\sigma_{\perp}$")
        plt.legend()
        plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
        plt.ylabel('$\sigma_{tilt\:jitter}$ [mas]', size=12)
        plt.xticks(size=11)
        plt.yticks(size=11)
        plt.ylim(0, 39)
        plt.title('Filtro K$_{s}$')

    def tiltJitter_Dither1_FilterJ_ExcludingCrowdness():
        imas = restoreObjectListFromFile('/home/gcarla/workspace/'
                                         '20161019/reduction_final/FilterJ'
                                         '/Dither1/sciJ_List.pkl')
        epsf = restoreObjectListFromFile('/home/gcarla/workspace/20161019/'
                                         'DataToRestore_forAnalysis/'
                                         'FilterJ/Dither1/epsfModels_list.pkl')

        def _fitStarsPositionExcludingCrowdedStars_dither1_filterJ(imaList):
            tt = tablesOfFittedStars_creator.createTablesListOfFittedStars(
                imaList)
            self = tt
            tt.epsfList = epsf
            tt.fitStarsPositionOnImasList()
            matchtab = tt.getListOfMatchingTabs()
            return matchtab

        def _maskCentralRegion_Imas_Dither1_FilterJ(imaList):
            for i in range(9):
                imaList[i].mask[600:1300, 700:1400] = True
            return imaList

        tabs = _fitStarsPositionExcludingCrowdedStars_dither1_filterJ(
            _maskCentralRegion_Imas_Dither1_FilterJ(imas))
        diff = differentialTJ_estimator.estimateDifferentialTJ(tabs, [414, 0])
        ae= diff.astrometricError()
        plt.figure()
        plot(diff.polCoord[0]*0.1178, ae[:, 0]*0.1178*1e03,
             '.', label="$\sigma_{\parallel}$")
        plt.legend()
        plot(diff.polCoord[0]*0.1178, ae[:, 1]*0.1178*1e03,
             '.', label="$\sigma_{\perp}$")
        plt.legend()
        plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
        plt.ylabel('$\sigma_{tilt\:jitter}$ [mas]', size=12)
        plt.xticks(size=11)
        plt.yticks(size=11)
        plt.title('Filtro J')


def matchStarsTablesList(tabsList, max_shift):
    match = match_astropy_tables.MatchTables(max_shift)
    refTab = match.match2TablesPhotometry(tabsList[0], tabsList[1])[0]
    for fitTab in tabsList[2:]:
        print('Matching tab %g with reference tab')
        _, refTab = match.match2TablesPhotometry(fitTab, refTab)
    matchingFitTabs = []
    for fitTab in tabsList:
        #print('Matching tab %g with reference tab' %fitTab)
        tab1, _ = match.match2TablesPhotometry(fitTab, refTab)
        matchingFitTabs.append(tab1)
    return matchingFitTabs


def matchStarsTablesListWithRefTab(tabsList, refTab, max_shift):
    match = match_astropy_tables.MatchTables(max_shift)
    #refTab = match.match2TablesPhotometry(tabsList[0], tabsList[1])[0]
    for fitTab in tabsList:
        _, refTab = match.match2TablesPhotometry(fitTab, refTab)
    matchingFitTabs = []
    for fitTab in tabsList:
        tab1, _ = match.match2TablesPhotometry(fitTab, refTab)
        matchingFitTabs.append(tab1)
    return matchingFitTabs


def main190312():

    epsf = restoreObjectListFromFile('/home/gcarla/workspace/20161019/'
                                     'DataToRestore_forAnalysis/'
                                     'FilterJ/Dither1/epsfModels_list.pkl')
    imas = restoreObjectListFromFile('/home/gcarla/workspace/'
                                     '20161019/reduction_final/FilterJ'
                                     '/Dither1/sciJ_List.pkl')
    ima_fit = image_fitter.ImageFitter(
        8e03, 3., 3, 0.1, 2.0, -1.0, 1.0, 5e04)
    fitTabs = []
    for i in range(len(imas)):
        ima_fit.fitStarsWithBasicPhotometry(imas[i], epsf[i],
                                            (21, 21), 45)
        fitTabs.append(ima_fit.getFitTable())
    matchingTabs = matchStarsTablesList(fitTabs)

    diff = differentialTJ_estimator.estimateDifferentialTJ(matchingTabs,
                                                           [414, 0])
    ae = diff.astrometricError()
    plt.figure()
    plot(diff.polCoord[0]*0.1178, ae[:, 0]*0.1178*1e03,
         '.', label="$\sigma_{\parallel}$")
    plt.legend()
    plot(diff.polCoord[0]*0.1178, ae[:, 1]*0.1178*1e03,
         '.', label="$\sigma_{\perp}$")
    plt.legend()
    plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
    plt.ylabel('$\sigma_{tilt\:jitter}$ [mas]', size=12)
    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.title('Filtro J')


def compareOurFitWithIDLStarFinder():
    tabs = restoreObjectListFromFile('/home/gcarla/workspace/'
                                     '20161019/DataToRestore_forAnalysis'
                                     '/FilterJ/Dither1/matchingTabs_list_J'
                                     '_dither1.pkl')
    arr = np.loadtxt('/home/gcarla/workspace/20161019/StarFinder'
                     '/starsList_J_1.txt')
    xPos = arr[:, 0]
    yPos = arr[:, 1]
    tabStarFinder = tabs[0].copy()
    tabStarFinder.remove_rows(range(len(tabStarFinder)))
    for i in range(1964):
        tabStarFinder.add_row()
    for i in range(len(tabStarFinder)):
        tabStarFinder['x_fit'][i] = xPos[i]
        tabStarFinder['y_fit'][i] = yPos[i]

    mm = match_astropy_tables.MatchTables(1)
    m1, m2 = mm.match2TablesPhotometry(tabs[0], tabStarFinder)
    plot(m1['x_fit']-m2['x_fit'], label='$\Delta$x')
    plt.legend()
    plot(m1['y_fit']-m2['y_fit'], label='$\Delta$y')
    plt.legend()
    plt.xticks(size=11)
    plt.yticks(size=11)
    return m1, m2


def testOurFittingMethodOnGaussianSourcesImageNoNoise():

    def _createImage():
        ima_cre = image_creator.ImageCreator((300, 300))
        ima = ima_cre.createMultipleGaussian(
            stddevXRange=[1.3, 1.3],
            fluxInPhotons=[1e05, 1e05],
            nStars=40)
        mm = np.zeros(ima.shape)
        mask = np.clip(mm, 0, 1)
        ima_ma = np.ma.masked_array(ima, mask=mask)
        return ima_ma, np.array(ima_cre._table['x_mean']), \
            np.array(ima_cre._table['y_mean'])

    def _buildEPSF(ima):
        epsf = ePSF_builder.ePSFBuilder(ima, 1e03, 3., size=20)
        epsf.extractStars()
        epsf.selectGoodStars()
        epsf.buildEPSF()
        model = epsf.getEPSFModel()
#         epsf_fit = image_fitter.ImageFitter(0.05, 3., 3., 0.1, 2.0, -1.0, 1.0)
#         epsf_fit.fitSingleStarWithGaussianFit(model.data)
#         #fitEpsf = epsf_fit.getFitParameters()
        return model

    ima, xTrue, yTrue = _createImage()
    model = _buildEPSF(ima)

    ima_fit = image_fitter.ImageFitter(1e03, 3., 1., 0.1, 2.0, -1.0, 1.0)
    ima_fit.fitStarsWithBasicPhotometry(ima, model, (21, 21), 20)
    fitTab = ima_fit.getFitTable()
    trueTab = fitTab.copy()
    trueTab.remove_rows(range(len(trueTab)))
    for i in range(xTrue.shape[0]):
        trueTab.add_row()
    for i in range(xTrue.shape[0]):
        trueTab['x_fit'][i] = xTrue[i]
    for i in range(yTrue.shape[0]):
        trueTab['y_fit'][i] = yTrue[i]
    match = match_astropy_tables.MatchTables(1)
    mTrue, mFit = match.match2TablesPhotometry(trueTab, fitTab)
    diffX = mTrue['x_fit']-mFit['x_fit']
    diffY = mTrue['y_fit']-mFit['y_fit']
    plt.figure()
    showNorm(ima)
    plt.figure()
    showNorm(np.minimum(model.data, 0.001))
    plt.figure()
    showNorm(model.data)
    plt.figure()
    plot(diffX)
    plot(diffY)
    return ima, mTrue, mFit, diffX, diffY


def testOurFittingMethodOnGaussianSourcesImageNoNoiseIntPx():

    def _createImage():
        ima_cre = image_creator.ImageCreator((300, 300))
        ima = ima_cre.createMultipleGaussianIntegerCentroids(
            stddevXRange=[1.3, 1.3],
            fluxInPhotons=[1e05, 1e05],
            nStars=50)
        mm = np.zeros(ima.shape)
        mask = np.clip(mm, 0, 1)
        ima_ma = np.ma.masked_array(ima, mask=mask)
        return ima_ma, np.array(ima_cre._table['x_mean']), \
            np.array(ima_cre._table['y_mean'])

    def _buildEPSF(ima):
        epsf = ePSF_builder.ePSFBuilder(ima, 1e03, 3., size=20)
        epsf.extractStars()
        epsf.selectGoodStars()
        epsf.buildEPSF()
        model = epsf.getEPSFModel()
#         epsf_fit = image_fitter.ImageFitter(0.05, 3., 3., 0.1, 2.0, -1.0, 1.0)
#         epsf_fit.fitSingleStarWithGaussianFit(model.data)
#         #fitEpsf = epsf_fit.getFitParameters()
        return model

    ima, xTrue, yTrue = _createImage()
    model = _buildEPSF(ima)

    ima_fit = image_fitter.ImageFitter(1e03, 3., 1., 0.1, 2.0, -1.0, 1.0)
    ima_fit.fitStarsWithBasicPhotometry(ima, model, (21, 21), 20)
    fitTab = ima_fit.getFitTable()
    trueTab = fitTab.copy()
    trueTab.remove_rows(range(len(trueTab)))
    for i in range(xTrue.shape[0]):
        trueTab.add_row()
    for i in range(xTrue.shape[0]):
        trueTab['x_fit'][i] = xTrue[i]
    for i in range(yTrue.shape[0]):
        trueTab['y_fit'][i] = yTrue[i]
    match = match_astropy_tables.MatchTables(1)
    mTrue, mFit = match.match2TablesPhotometry(trueTab, fitTab)
    diffX = mTrue['x_fit']-mFit['x_fit']
    diffY = mTrue['y_fit']-mFit['y_fit']
    plt.figure()
    showNorm(ima)
    plt.figure()
    showNorm(np.minimum(model.data, 0.001))
    plt.figure()
    plot(diffX)
    plot(diffY)
    return ima, mTrue, mFit, diffX, diffY

#     def _centroid(ima):
#         sy, sx= ima.shape
#         y, x= np.mgrid[0:sy, 0:sx]
#         cx=np.sum(ima*x)/np.sum(ima)
#         cy=np.sum(ima*y)/np.sum(ima)
#         return cx, cy
#
#     _centroid(model.data)


def matchAllFiltersTabs(tabJ, tabH, tabK):

    mm = matchStarsTablesList([tabJ[0], tabH[0], tabK[0]],
                              max_shift=5)
    refJ = mm[0]
    refH = mm[1]
    refK = mm[2]
    newTabJ = matchStarsTablesListWithRefTab(tabJ, refJ, 1)
    newTabH = matchStarsTablesListWithRefTab(tabH, refH, 1)
    newTabK = matchStarsTablesListWithRefTab(tabK, refK, 1)
    return newTabJ, newTabH, newTabK


def main190320_matchAllFiltersTabs():
    tabJ1 = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterJ/'
        'Dither1/matchingTabs_list_J_dither1.pkl')
    tabH1 = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterH/'
        'Dither1/matchingTabs_list_H_dither1.pkl')
    tabK1 = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterKs/'
        'Dither1/matchingTabs_list_Ks_dither1.pkl')

    mm = matchStarsTablesList([tabJ1[0], tabH1[0], tabK1[0]],
                              max_shift=5)
    refJ = mm[0]
    refH = mm[1]
    refK = mm[2]
    newTabJ1 = matchStarsTablesListWithRefTab(tabJ1, refJ, 1)
    newTabH1 = matchStarsTablesListWithRefTab(tabH1, refH, 1)
    newTabK1 = matchStarsTablesListWithRefTab(tabK1, refK, 1)
    return newTabJ1, newTabH1, newTabK1


def main190320_J1():
    fitAcc4 = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_4_HalfPx.pkl')
    photErr4_inMas = np.sqrt(fitAcc4.errXList+fitAcc4.errYList)*0.119*1e03

    tabJ1 = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterJ/'
        'Dither1/matchingTabs_list_J_dither1.pkl')
    aeJ1 = astrometricError_estimator.EstimateAstrometricError(tabJ1)
    aeJ1.createCubeOfStarsInfo()
    astroErrors_J1 = aeJ1.getStandardAstrometricErrorinArcsec()*1e03
    fluxes_J1 = aeJ1.getStarsFlux().mean(axis=0)
    plt.semilogx(fluxes_J1, astroErrors_J1, '.')
    plt.semilogx(fitAcc4.fluxVector, photErr4_inMas)


def main190320_H1():
    fitAcc3 = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_3_HalfPx.pkl')
    photErr3_inMas = np.sqrt(fitAcc3.errXList+fitAcc3.errYList)*0.119*1e03

    tabH1 = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterH/'
        'Dither1/matchingTabs_list_H_dither1.pkl')
    aeH1 = astrometricError_estimator.EstimateAstrometricError(tabH1)
    aeH1.createCubeOfStarsInfo()
    astroErrors_H1 = aeH1.getStandardAstrometricErrorinArcsec()*1e03
    fluxes_H1 = aeH1.getStarsFlux().mean(axis=0)
    plt.semilogx(fluxes_H1, astroErrors_H1, '.')
    plt.semilogx(fitAcc3.fluxVector, photErr3_inMas)


def main190320_K1():
    fitAcc2 = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_2_HalfPx.pkl')
    photErr2_inMas = np.sqrt(fitAcc2.errXList+fitAcc2.errYList)*0.119*1e03

    tabK1 = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterKs/'
        'Dither1/matchingTabs_list_Ks_dither1.pkl')
    aeK1 = astrometricError_estimator.EstimateAstrometricError(tabK1)
    aeK1.createCubeOfStarsInfo()
    astroErrors_K1 = aeK1.getStandardAstrometricErrorinArcsec()*1e03
    fluxes_K1 = aeK1.getStarsFlux().mean(axis=0)
    plt.semilogx(fluxes_K1, astroErrors_K1, '.')
    plt.semilogx(fitAcc2.fluxVector, photErr2_inMas)


def main190320_fluxesJ_onField():
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterJ/'
        'Dither1/matchingTabs_list_J_dither1.pkl')
    ae = astrometricError_estimator.EstimateAstrometricError(tab)
    ae.createCubeOfStarsInfo()
    meanX = (ae.getMeanPositionX() - 1024)*0.119
    meanY = (ae.getMeanPositionY() - 1024)*0.119
    fluxes = ae.getStarsFlux().mean(axis=0)
    plt.figure()
    plt.scatter(meanX, meanY, c=fluxes, s=40, norm=LogNorm())
    plt.xlim(-120, 120)
    plt.ylim(-120, 120)
    cb = plt.colorbar()
    cb.set_label(label='Flusso [fotoni]', size=12)
    cb.ax.tick_params(labelsize=11)
    plt.xlabel('arcsec', size=12)
    plt.ylabel('arcsec', size=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    return fluxes


def main190320_fluxesH_onField():
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterH/'
        'Dither1/matchingTabs_list_H_dither1.pkl')
    ae = astrometricError_estimator.EstimateAstrometricError(tab)
    ae.createCubeOfStarsInfo()
    meanX = (ae.getMeanPositionX() - 1024)*0.119
    meanY = (ae.getMeanPositionY() - 1024)*0.119
    fluxes = ae.getStarsFlux().mean(axis=0)
    plt.figure()
    plt.scatter(meanX, meanY, c=fluxes, s=40, norm=LogNorm())
    plt.xlim(-120, 120)
    plt.ylim(-120, 120)
    cb = plt.colorbar()
    cb.set_label(label='Flusso [fotoni]', size=12)
    cb.ax.tick_params(labelsize=11)
    plt.xlabel('arcsec', size=12)
    plt.ylabel('arcsec', size=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    return fluxes


def main190320_fluxesK_onField():
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterKs/'
        'Dither1/matchingTabs_list_Ks_dither1.pkl')
    ae = astrometricError_estimator.EstimateAstrometricError(tab)
    ae.createCubeOfStarsInfo()
    meanX = (ae.getMeanPositionX() - 1024)*0.119
    meanY = (ae.getMeanPositionY() - 1024)*0.119
    fluxes = ae.getStarsFlux().mean(axis=0)
    plt.figure()
    plt.scatter(meanX, meanY, c=fluxes, s=40, norm=LogNorm())
    plt.xlim(-120, 120)
    plt.ylim(-120, 120)
    cb = plt.colorbar()
    cb.set_label(label='Flusso [fotoni]', size=12)
    cb.ax.tick_params(labelsize=11)
    plt.xlabel('arcsec', size=12)
    plt.ylabel('arcsec', size=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    return fluxes


def main190320_TT_J_parallel_colorByFlux():
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterJ/'
        'Dither1/matchingTabs_list_J_dither1.pkl')
    de = differentialTJ_estimator.estimateDifferentialTJ(tab, (418, 1375))
    err = de.astrometricError()
    coord = de.polCoord[0]*0.119
    errPara = err[:, 0]*119
    fluxes = de.est.getStarsFlux().mean(axis=0)
    plt.scatter(coord, errPara, s=20, c=fluxes, norm=LogNorm())
    plt.colorbar(label='Flusso')
    plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
    plt.ylabel('$\sigma_{\parallel}$ [mas]', size=12)
    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.ylim(0, 39)


def main190320_TT_H_parallel_colorByFlux():
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterH/'
        'Dither1/matchingTabs_list_H_dither1.pkl')
    de = differentialTJ_estimator.estimateDifferentialTJ(tab, (418, 1375))
    err = de.astrometricError()
    coord = de.polCoord[0]*0.119
    errPara = err[:, 0]*119
    fluxes = de.est.getStarsFlux().mean(axis=0)
    plt.scatter(coord, errPara, s=20, c=fluxes, norm=LogNorm())
    plt.colorbar(label='Flusso')
    plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
    plt.ylabel('$\sigma_{\parallel}$ [mas]', size=12)
    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.ylim(0, 39)


def main190320_TT_K_parallel_colorByFlux():
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterKs/'
        'Dither1/matchingTabs_list_Ks_dither1.pkl')
    de = differentialTJ_estimator.estimateDifferentialTJ(tab, (416, 1373))
    err = de.astrometricError()
    coord = de.polCoord[0]*0.119
    errPara = err[:, 0]*119
    fluxes = de.est.getStarsFlux().mean(axis=0)
    plt.scatter(coord, errPara, s=20, c=fluxes, norm=LogNorm())
    plt.colorbar(label='Flusso')
    plt.xlabel('d$_{NGS}$ [arcsec]', size=12)
    plt.ylabel('$\sigma_{\parallel}$ [mas]', size=12)
    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.ylim(0, 39)


def main190320_TT_J_FindBumpIndices():
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterJ/'
        'Dither1/matchingTabs_list_J_dither1.pkl')

    de = differentialTJ_estimator.estimateDifferentialTJ(tab, (418, 1375))
    tjErr = de.astrometricError()
    sPara = tjErr[:, 0]*119
    coord = de.polCoord[0]*0.119
    ii = np.argwhere((np.abs(coord-100) < 15) & ((25-sPara) < 5))
    meanPos = de.est.getMeanPosition()
    plt.scatter((meanPos[0, :]-1024)*0.119, (meanPos[1, :]-1024)*0.119)
    xC = meanPos[0, :][ii]
    yC = meanPos[1, :][ii]
    plt.scatter((xC-1024)*0.119, (yC-1024)*0.119)
    plt.xlim(-120, 120)
    plt.ylim(-120, 120)


def main190321PlotFitAccuracyIntPx():
    acc2Int = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_2_IntPx.pkl')
    acc3Int = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_3_IntPx.pkl')
    acc4Int = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_4_IntPx.pkl')
    acc5Int = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_5_IntPx.pkl')

    plt.loglog(acc2Int.fluxVector,
               np.sqrt(acc2Int.errXList+acc2Int.errYList)*119, 'r',
               label='FWHM = %g $^{\prime\prime}$' %(acc2Int._fwhm*0.119))
    plt.loglog(acc3Int.fluxVector,
               np.sqrt(acc3Int.errXList+acc3Int.errYList)*119, 'b',
               label='FWHM = %g $^{\prime\prime}$' %(acc3Int._fwhm*0.119))
    plt.loglog(acc4Int.fluxVector,
               np.sqrt(acc4Int.errXList+acc4Int.errYList)*119, 'y',
               label='FWHM = %g $^{\prime\prime}$' %(acc4Int._fwhm*0.119))
    plt.loglog(acc5Int.fluxVector,
               np.sqrt(acc5Int.errXList+acc5Int.errYList)*119, 'g',
               label='FWHM = %g $^{\prime\prime}$' %(acc5Int._fwhm*0.119))
    plt.legend()
    plt.xlabel('N [fotoni]', size=12)
    plt.ylabel('rms [mas]', size=12)
    plt.xticks(size=11)
    plt.yticks(size=11)


def main190321PlotFitAccuracyHalfPx():
    acc2Half = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_2_HalfPx.pkl')
    acc3Half = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_3_HalfPx.pkl')
    acc4Half = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_4_HalfPx.pkl')
    acc5Half = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_5_HalfPx.pkl')

    plt.loglog(acc2Half.fluxVector,
               np.sqrt(acc2Half.errXList+acc2Half.errYList)*119, 'r',
               label='FWHM = %g $^{\prime\prime}$' %(acc2Half._fwhm*0.119))
    plt.loglog(acc3Half.fluxVector,
               np.sqrt(acc3Half.errXList+acc3Half.errYList)*119, 'b',
               label='FWHM = %g $^{\prime\prime}$' %(acc3Half._fwhm*0.119))
    plt.loglog(acc4Half.fluxVector,
               np.sqrt(acc4Half.errXList+acc4Half.errYList)*119, 'y',
               label='FWHM = %g $^{\prime\prime}$' %(acc4Half._fwhm*0.119))
    plt.loglog(acc5Half.fluxVector,
               np.sqrt(acc5Half.errXList+acc5Half.errYList)*119, 'g',
               label='FWHM = %g $^{\prime\prime}$' %(acc5Half._fwhm*0.119))
    plt.legend()
    plt.xlabel('N [fotoni]', size=12)
    plt.ylabel('rms [mas]', size=12)
    plt.xticks(size=11)
    plt.yticks(size=11)


def main190321PlotFitAccuracyWithinPixelWithFWHM2Px():
    acc0 = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_2_diagPx0.pkl')
    acc1 = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_2_diagPx1.pkl')
    acc2 = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_2_diagPx2.pkl')
    acc3 = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_2_diagPx3.pkl')
    acc4 = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_2_diagPx4.pkl')
    acc5 = restoreObjectListFromFile(
        '/home/gcarla/workspace/fit_test/test_fitAccuracy/fwhm_2_diagPx5.pkl')

    plt.loglog(acc0.fluxVector, np.sqrt(acc0.errXList+acc0.errYList)*119, 'r',
               label='x = %g, y = %g' %(acc0._posX, acc0._posY))
    plt.loglog(acc0.fluxVector, np.sqrt(acc1.errXList+acc1.errYList)*119, 'b',
               label='x = %g, y = %g' %(acc1._posX, acc1._posY))
    plt.loglog(acc0.fluxVector, np.sqrt(acc2.errXList+acc2.errYList)*119, 'y',
               label='x = %g, y = %g' %(acc2._posX, acc2._posY))
    plt.loglog(acc0.fluxVector, np.sqrt(acc3.errXList+acc3.errYList)*119, 'g',
               label='x = %g, y = %g' %(acc3._posX, acc3._posY))
    plt.loglog(acc0.fluxVector, np.sqrt(acc4.errXList+acc4.errYList)*119, 'm',
               label='x = %g, y = %g' %(acc4._posX, acc4._posY))
    plt.loglog(acc0.fluxVector, np.sqrt(acc5.errXList+acc5.errYList)*119, 'c',
               label='x = %g, y = %g' %(acc5._posX, acc5._posY))
    plt.legend()
    plt.xlabel('N [fotoni]', size=12)
    plt.ylabel('rms [mas]', size=12)
    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.title('FWHM = 0.238$^{\prime\prime}$')


class testFitAccuracyWithGaussianFit():

    def __init__(self,
                 shape=(50, 50),
                 posX=23.77,
                 posY=17.01,
                 fwhm=1):
        self._shape = shape
        self._posX = posX
        self._posY = posY
        self._fwhm = fwhm
        self._stdX = gaussian_fwhm_to_sigma*fwhm
        self._stdY = gaussian_fwhm_to_sigma*fwhm

    def _createSetOfSameGaussianSource(self,
                                       flux,
                                       howManyImages=100):
        ima_cre = image_creator.ImageCreator(self._shape)
        ima_cre.usePoissonNoise(True)
        self.imaSet = []
        print("Create %d images" % howManyImages)
        for i in range(howManyImages):
            ima = ima_cre.createGaussianImage(self._posX, self._posY,
                                              flux, self._stdX, self._stdY)
            self.imaSet.append(np.ma.masked_array(ima))
        return self.imaSet

#     def _getEPSFforSetOfSameGaussian(self, flux):
#         epsfList = []
#         for ima in self._createSetOfSameGaussianSource(flux):
#             builder = ePSF_builder.ePSFBuilder(ima, threshold=800, fwhm=3.,
#                                                size=20, peakmax=None)
#             builder.extractStars()
#             builder.buildEPSF()
#             epsfModel = builder.getEPSFModel()
#             epsfList.append(epsfModel)
#         return epsfList
#
#     def _fitStarsInSetOfSameGaussian(self, flux):
#         threshold = 0.1*flux/(2*pi*self._stdX*self._stdY)
#         fwhm = 0.7*gaussian_sigma_to_fwhm*self._stdX
#         imas = self._createSetOfSameGaussianSource(flux)
#         epsfs = self._getEPSFforSetOfSameGaussian(flux)
#         ima_fit = image_fitter.ImageFitter(thresholdInPhotons=threshold,
#                                            fwhm=fwhm, min_separation=3.,
#                                            sharplo=0.1, sharphi=2.0,
#                                            roundlo=-1.0, roundhi=1.0,
#                                            peakmax=None)
#         for i in range(len(imas)):
#             ima_fit.fitStarsWithBasicPhotometry(image=imas[i],
#                                                 model=epsfs[i],
#                                                 fitshape=(11,11),
#                                                 apertureRadius=10)
#             fitTab = ima_fit.getFitTable()
#             self._posX - fitTab['x_fit']

#     def _getEPSFOnSingleFrame(self, ima):
#         print("Building EPSF")
#         builder = ePSF_builder.ePSFBuilder(ima, threshold=80, fwhm=self._fwhm,
#                                            size=20, peakmax=None)
#         builder.extractStars()
#         builder.buildEPSF()
#         epsfModel = builder.getEPSFModel()
#         return epsfModel

    def _fitStarWithEPSFOnSingleFrame(self,
                                      ima,
                                      flux):
        print("Fitting star")
        threshold = 0.1*flux/(2*pi*self._stdX*self._stdY)
        fwhm = self._fwhm
        ima_fit = image_fitter.ImageFitter(thresholdInPhotons=threshold,
                                           fwhm=fwhm, min_separation=3.,
                                           sharplo=0.1, sharphi=2.0,
                                           roundlo=-1.0, roundhi=1.0,
                                           peakmax=None)
        ima_fit.fitSingleStarWithGaussianFit(image=ima)
        centr = ima_fit.getCentroid()
        return centr

    def _fitAllFrames(self, flux):
        self._fitTabs = []
        imasList = self._createSetOfSameGaussianSource(flux)
        print("Fitting images")
        for ima in imasList:
            centr = self._fitStarWithEPSFOnSingleFrame(ima,
                                                       flux)
            self._fitTabs.append(centr)
        return self._fitTabs

    def _measurePositionErrorForOneSetOfFrames(self, flux):
        tabs = self._fitAllFrames(flux)
        dxList = []
        dyList = []
        for tab in tabs:
            dx = self._posX - tab[0]
            dy = self._posY - tab[1]
            dxList.append(dx)
            dyList.append(dy)
        errX = np.array(dxList)
        errY = np.array(dyList)
        return tabs, np.sum(errX**2)/len(tabs), np.sum(errY**2)/len(tabs)

#     def _estimateAstrometricError(self, flux, sx, sy):
#         tabs = self._fitAllFrames(flux, sx, sy)
#         estimator = astrometricError_estimator.EstimateAstrometricError(tabs)
#         estimator.createCubeOfStarsInfo()
#         astromError = estimator.getStandardAstrometricErrorinPixels()
#         return astromError

#     def _buildModelAtHighFlux(self):
#         highFlux=1e12
#         ima = self._createSetOfSameGaussianSource(highFlux,
#                                                   howManyImages=1)[0]
#         return self._getEPSFOnSingleFrame(ima)

    def measurePositionErrorForDifferentFluxes(self,
                                               fluxVector=None):

        if fluxVector is None:
            fluxVector=np.logspace(3, 8, 30)
        #self._model = self._buildModelAtHighFlux()
        resX = []
        resY = []
        self.allTabs = []
        self.fluxVector= fluxVector
        for flux in self.fluxVector:
            print("Computing error for flux %g" % flux)
            tabs, errX, errY = self._measurePositionErrorForOneSetOfFrames(
                flux)
            self.allTabs.append(tabs)
            resX.append(errX)
            resY.append(errY)
        self.errXList=np.array(resX)
        self.errYList=np.array(resY)

    def plot(self, color, scale):
        import matplotlib.pyplot as plt
        if scale=='px':
            plt.loglog(
                self.fluxVector, self.errXList, label='FWHM = %g px' % self._fwhm,
                color=color)
            plt.loglog(
                self.fluxVector, self.errYList, '-.', color=color)
            plt.xlabel('PSF Flux [phot]')
            plt.ylabel('Mean Square Error [px]')
            plt.legend()
        elif scale=='mas':
            plt.loglog(
                self.fluxVector, self.errXList*0.119*1e03,
                label='FWHM = %g $^{\prime\prime}$' % (self._fwhm*0.119),
                color=color)
            plt.loglog(
                self.fluxVector, self.errYList*0.119*1e03, '-.', color=color)
            plt.xlabel('PSF Flux [phot]')
            plt.ylabel('Mean Square Error [mas]')
            plt.legend()


def main190325_getDisplacementsInImasOpenLoop():
    openTab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/openLoop/matchingTabs_list_J_openLoop.pkl')

    aeOpen = astrometricError_estimator.EstimateAstrometricError(openTab)
    aeOpen.createCubeOfStarsInfo()
    dx = []
    dy = []
    for i in range(len(openTab)):
        dispx, dispy = aeOpen.getDisplacementsFromMeanPositions(i)
        dx.append(dispx)
        dy.append(dispy)


def main190325_plotDisplacementsInImasOpenLoop():
    xMean = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/dataToRestore_forPlot/'
        'FilterJ_openloop/DisplacementsFromMean/stars_meanPosX.pkl')
    yMean = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/dataToRestore_forPlot/'
        'FilterJ_openloop/DisplacementsFromMean/stars_meanPosY.pkl')
    dx_list = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/dataToRestore_forPlot/'
        'FilterJ_openloop/DisplacementsFromMean/'
        'displacementsXFromMeanPos_list.pkl')
    dy_list = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/dataToRestore_forPlot/'
        'FilterJ_openloop/DisplacementsFromMean/'
        'displacementsYFromMeanPos_list.pkl')

    for i in range(len(dx_list)):
        plt.figure()
        plotDisplacementsInLUCI1sFoV_multiColor(xMean, yMean,
                                                dx_list[i], dy_list[i])


def main190325_plotDifferentialTJWithAlignedNGSs():
    tabJ = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/Dither1/matchingTabsWithNGSalignedOnMeanNGS_list_J_dither1.pkl')
    tabH = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterH/Dither1/matchingTabsWithNGSalignedOnMeanNGS_list_H_dither1.pkl')
    tabK = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterKs/Dither1/matchingTabsWithNGSalignedOnMeanNGS_list_Ks_dither1.pkl')

    deJ = differentialTJ_estimator.estimateDifferentialTJ(tabJ, (418, 1375))
    deH = differentialTJ_estimator.estimateDifferentialTJ(tabH, (418, 1375))
    deK = differentialTJ_estimator.estimateDifferentialTJ(tabK, (415, 1373))

    deJ.plot()
    plt.title('J')
    plt.figure()
    deH.plot()
    plt.title('H')
    plt.figure()
    deK.plot()
    plt.title('K$_{s}$')


def main190325_exampleDifferentialTJCurveFitting():
    tabJ = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/Dither1/matchingTabsWithNGSalignedOnMeanNGS_list_J_dither1.pkl')
    deJ = differentialTJ_estimator.estimateDifferentialTJ(tabJ, (418, 1375))
    errJPara = deJ.astrometricError()[:, 0]*119
    errJPerp = deJ.astrometricError()[:, 1]*119
    thetaJ = deJ.polCoord[0]*0.119

    def _func(th, a):
        return a*th

    paramJPara, covJPara = curve_fit(_func, thetaJ, errJPara)
    paramJPerp, covJPerp = curve_fit(_func, thetaJ, errJPerp)

    return paramJPara, paramJPerp


def fitEPSFWithMoffat(epsfData, threshold=0.005, fwhm=3., minsep=10,
                      sharplo=0.1, sharphi=2.0, roundlo=-1., roundhi=1.):
    ima_fit = image_fitter.ImageFitter(threshold, fwhm, minsep,
                                       sharplo, sharphi, roundlo, roundhi)
    ima_fit.fitSingleStarWithMoffatFit(epsfData)

    alpha = ima_fit.getFitParameters().alpha[0]
    gamma = ima_fit.getFitParameters().gamma[0]
    fwhmMoffat = 2*gamma*np.sqrt(2**(1./alpha) - 1)
    return alpha, gamma, fwhmMoffat*0.119,


def GaussianFit(ima, threshold, fwhm, minsep,
                sharplo, sharphi, roundlo, roundhi):
    ima_fit = image_fitter.ImageFitter(threshold, fwhm, minsep,
                                       sharplo, sharphi, roundlo, roundhi)
    ima_fit.fitSingleStarWithGaussianFit(ima)
    fxInArcsec = ima_fit.getFitParameters().x_fwhm[0]*0.119
    fyInArcsec = ima_fit.getFitParameters().y_fwhm[0]*0.119
    theta = ima_fit.getFitParameters().theta[0]
    return fxInArcsec, fyInArcsec, theta


def main190326_plotTotalAstrometricErrorOnLUCIsField():
    tabJ = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/Dither1/matchingTabs_list_J_dither1.pkl')
    tabH = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterH/Dither1/matchingTabs_list_H_dither1.pkl')
    tabK = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterKs/Dither1/matchingTabs_list_Ks_dither1.pkl')

    def _initializeAstrometricErrorEstimatorClassAndPlot(tab):
        ae = astrometricError_estimator.EstimateAstrometricError(tab)
        ae.createCubeOfStarsInfo()
        plt.figure()
        ae.plotStandardAstroErrorOntheFieldInArcsec()
        #plt.clim(5, 35)

    _initializeAstrometricErrorEstimatorClassAndPlot(tabJ)
    plt.title('J')
    _initializeAstrometricErrorEstimatorClassAndPlot(tabH)
    plt.title('H')
    _initializeAstrometricErrorEstimatorClassAndPlot(tabK)
    plt.title('K$_{s}$')


def main190326_getAstrometricErrorLimits():
    tabJ = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/Dither1/matchingTabs_list_J_dither1.pkl')
    tabH = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterH/Dither1/matchingTabs_list_H_dither1.pkl')
    tabK = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterKs/Dither1/matchingTabs_list_Ks_dither1.pkl')

    def _initializeAstrometricErrorEstimatorClassAndGetErrors(tab):
        ae = astrometricError_estimator.EstimateAstrometricError(tab)
        ae.createCubeOfStarsInfo()
        err = ae.getStandardAstrometricErrorinArcsec()
        print('ErrMin, ErrMax = %g [mas], %g [mas]'
              %(err.min()*1e03, err.max()*1e03))
        return err

    def _plot(err, label):
        plot(err, label=label)
        plt.xlabel('N [stelle]', size=13)
        plt.ylabel('Errore astrometrico [mas]', size=13)
        plt.xticks(size=11)
        plt.yticks(size=11)

    print('Filter J:')
    errJ = _initializeAstrometricErrorEstimatorClassAndGetErrors(tabJ)
    print('Filter H:')
    errH = _initializeAstrometricErrorEstimatorClassAndGetErrors(tabH)
    print('Filter K$_{s}$:')
    errK = _initializeAstrometricErrorEstimatorClassAndGetErrors(tabK)

    _plot(errJ*1e03, label='J')
    _plot(errH*1e03, label='H')
    _plot(errK*1e03, label='K$_{s}$')
    plt.legend()


def findIndexOfStarsNearNgs(tab, xNgs, yNgs):
    xx = np.array(tab[0]['x_fit'])
    yy = np.array(tab[0]['y_fit'])
    i = np.argwhere(np.sqrt((xx-xNgs)**2+(yy-yNgs)**2) < 210)
    return i


def main190326_plotDisplacementsNearNgs():
    tabJ = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/Dither1/matchingTabs_list_J_dither1.pkl')
    tabH = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterH/Dither1/matchingTabs_list_H_dither1.pkl')
    tabK = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterKs/Dither1/matchingTabs_list_Ks_dither1.pkl')

    def _initializeAstrometricErrorClass(tab, idx):
        ae = astrometricError_estimator.EstimateAstrometricError(tab)
        ae.createCubeOfStarsInfo()
        xmean = ae.getMeanPositionX()[idx]
        ymean = ae.getMeanPositionY()[idx]
        dx, dy = ae.getDisplacementsFromMeanPositions(0)
        return xmean, ymean, dx[idx], dy[idx]

    xNgs, yNgs, dxNgs, dyNgs = _initializeAstrometricErrorClass(tabJ, 197)
    plotDisplacementsInLUCI1sFoV_multiColor(xNgs, yNgs, dxNgs, dyNgs)


def main190327_astromErrorOfStarsNearNgsVsPhotonNoise_J():
    tabJ = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/Dither1/matchingTabsWithNGSalignedOnMeanNGS_list_J_dither1.pkl')
    acc = restoreObjectListFromFile('/home/gcarla/workspace/fit_test/'
                                    'test_fitAccuracy/fwhm_4_IntPx.pkl')
    acc2 = restoreObjectListFromFile('/home/gcarla/workspace/fit_test/'
                                     'test_fitAccuracy/fwhm_4_HalfPx.pkl')

    def _getErrorsAndFluxes(tab, idx):
        ae = astrometricError_estimator.EstimateAstrometricError(tab)
        ae.createCubeOfStarsInfo()
        err = ae.getStandardAstrometricErrorinArcsec()[idx]*1e03
        flux = ae.getStarsFlux().mean(axis=0)[idx]
        return err, flux

    e1, f1 = _getErrorsAndFluxes(tabJ, 177)
    e2, f2 = _getErrorsAndFluxes(tabJ, 181)
    e3, f3 = _getErrorsAndFluxes(tabJ, 196)
    e4, f4 = _getErrorsAndFluxes(tabJ, 202)
    e5, f5 = _getErrorsAndFluxes(tabJ, 203)
    e6, f6 = _getErrorsAndFluxes(tabJ, 204)

    errors = np.array([e1, e2, e3, e4, e5])
    fluxes = np.array([f1, f2, f3, f4, f5])

    plt.loglog(acc.fluxVector, np.sqrt(acc.errXList+acc.errYList)*119,
               label='Errore fotonico, (x,y) = .0 px')
    plt.loglog(acc.fluxVector, np.sqrt(acc2.errXList+acc2.errYList)*119,
               label='Errore fotonico, (x,y) = .5 px')
    plt.loglog(fluxes, errors, '.', label='Errore residuo delle stelle vicine'
               ' alla NGS dopo allineamento')
    plt.legend()


def main190327_astromErrorOfStarsNearNgsVsPhotonNoise_H():
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterH/Dither1/matchingTabsWithNGSalignedOnMeanNGS_list_H_dither1.pkl')
    acc = restoreObjectListFromFile('/home/gcarla/workspace/fit_test/'
                                    'test_fitAccuracy/fwhm_3_IntPx.pkl')
    acc2 = restoreObjectListFromFile('/home/gcarla/workspace/fit_test/'
                                     'test_fitAccuracy/fwhm_3_HalfPx.pkl')

    def _getErrorsAndFluxes(table, idx):
        ae = astrometricError_estimator.EstimateAstrometricError(tab)
        ae.createCubeOfStarsInfo()
        err = ae.getStandardAstrometricErrorinArcsec()[idx]*1e03
        flux = ae.getStarsFlux().mean(axis=0)[idx]
        return err, flux

    e1, f1 = _getErrorsAndFluxes(tab, 177)
    e2, f2 = _getErrorsAndFluxes(tab, 181)
    e3, f3 = _getErrorsAndFluxes(tab, 196)
    e4, f4 = _getErrorsAndFluxes(tab, 202)
    e5, f5 = _getErrorsAndFluxes(tab, 203)
    e6, f6 = _getErrorsAndFluxes(tab, 204)

    errors = np.array([e1, e2, e3, e4, e5])
    fluxes = np.array([f1, f2, f3, f4, f5])

    plt.loglog(acc.fluxVector, np.sqrt(acc.errXList+acc.errYList)*119,
               label='Errore fotonico, (x,y) = .0 px')
    plt.loglog(acc.fluxVector, np.sqrt(acc2.errXList+acc2.errYList)*119,
               label='Errore fotonico, (x,y) = .5 px')
    plt.loglog(fluxes, errors, '.', label='Errore residuo delle stelle vicine'
               ' alla NGS dopo allineamento')
    plt.legend()


def main190327_astromErrorOfStarsNearNgsVsPhotonNoise_K():
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterKs/Dither1/matchingTabsWithNGSalignedOnMeanNGS_list_Ks_dither1.pkl')
    acc = restoreObjectListFromFile('/home/gcarla/workspace/fit_test/'
                                    'test_fitAccuracy/fwhm_2_IntPx.pkl')
    acc2 = restoreObjectListFromFile('/home/gcarla/workspace/fit_test/'
                                     'test_fitAccuracy/fwhm_2_HalfPx.pkl')

    def _getErrorsAndFluxes(table, idx):
        ae = astrometricError_estimator.EstimateAstrometricError(tab)
        ae.createCubeOfStarsInfo()
        err = ae.getStandardAstrometricErrorinArcsec()[idx]*1e03
        flux = ae.getStarsFlux().mean(axis=0)[idx]
        return err, flux

    e1, f1 = _getErrorsAndFluxes(tab, 177)
    e2, f2 = _getErrorsAndFluxes(tab, 181)
    e3, f3 = _getErrorsAndFluxes(tab, 196)
    e4, f4 = _getErrorsAndFluxes(tab, 202)
    e5, f5 = _getErrorsAndFluxes(tab, 203)
    e6, f6 = _getErrorsAndFluxes(tab, 204)

    errors = np.array([e1, e2, e3, e4, e5])
    fluxes = np.array([f1, f2, f3, f4, f5])

    plt.loglog(acc.fluxVector, np.sqrt(acc.errXList+acc.errYList)*119,
               label='Errore fotonico, (x,y) = .0 px')
    plt.loglog(acc.fluxVector, np.sqrt(acc2.errXList+acc2.errYList)*119,
               label='Errore fotonico, (x,y) = .5 px')
    plt.loglog(fluxes, errors, '.', label='Errore residuo delle stelle vicine'
               ' alla NGS dopo allineamento')
    plt.legend()


def getStarsMagnitudes(tab, tabIdxRefStar, refMag):
    ae = astrometricError_estimator.EstimateAstrometricError(tab)
    ae.createCubeOfStarsInfo()
    fluxes = ae.getStarsFlux().mean(axis=0)
    refFlux = fluxes[tabIdxRefStar]
    allStarsMag = -2.5*np.log10(fluxes/refFlux) + refMag
    return fluxes, allStarsMag


def main190327_plotMagnitudesVsFluxes():
    tabJ = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/Dither1/matchingTabs_list_J_dither1.pkl')
    tabH = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterH/Dither1/matchingTabs_list_H_dither1.pkl')
    tabK = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterKs/Dither1/matchingTabs_list_Ks_dither1.pkl')

    fluxesJ, magJ = getStarsMagnitudes(tabJ, 197, 13.537)
    fluxesH, magH = getStarsMagnitudes(tabH, 197, 13.344)
    fluxesK, magK = getStarsMagnitudes(tabK, 197, 13.232)

    def _plot(fluxs, mags, filterStr):
        plt.semilogx(fluxs, mags, '.')
        plt.xlabel('Flusso [fotoni]', size=12)
        plt.xticks(size=11)
        plt.ylabel('mag'+filterStr, size=12)
        plt.yticks(size=11)

    _plot(fluxesJ, magJ, 'J')
    plt.figure()
    _plot(fluxesH, magH, 'H')
    plt.figure()
    _plot(fluxesK, magK, 'K$_{s}$')


def main190327_differentialTJPlotOnlyBrighterStars():
    tabJ1 = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/Dither1/matchingTabs_list_J_dither1.pkl')
    aeJ = astrometricError_estimator.EstimateAstrometricError(tabJ1)
    aeJ.createCubeOfStarsInfo()
    fluxesJ = aeJ.getStarsFlux().mean(axis=0)
    i = np.argwhere(fluxesJ > 1e05)
    tabsHighFlux = []
    for tab in tabJ1:
        tabsHighFlux.append(tab[i])
    deJ = differentialTJ_estimator.estimateDifferentialTJ(tabsHighFlux,
                                                          (418, 1375))

    deJ.plot()


def main190327_differentialTJOnImasOpenLoopAfterAlignment():
    tabsOpen = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/FilterJ/'
        'openLoop/matchingTabs_list_J_openLoop.pkl')
    align = ngs_aligner.alignNGS(tabsOpen, (418, 1375), 10)
    align.alignCoordsOnMeanNGS()
    newtabsOpen = align.getNewStarsTabsList()
    de = differentialTJ_estimator.estimateDifferentialTJ(newtabsOpen,
                                                         (418, 1375), 15)
    de.plot()


def main190327_plotAndSaveDisplacementsInImasCloseLoop_J(w=0.005):
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterJ/Dither1/matchingTabs_list_J_dither1.pkl')
    ae = astrometricError_estimator.EstimateAstrometricError(tab)
    ae.createCubeOfStarsInfo()

    xMean = ae.getMeanPositionX()
    yMean = ae.getMeanPositionY()
    dx_list = []
    dy_list = []
    for i in range(len(tab)):
        dx, dy = ae.getDisplacementsFromMeanPositions(i)
        dx_list.append(dx)
        dy_list.append(dy)

    for i in range(len(dx_list)):
        # plt.figure()
        plotDisplacementsInLUCI1sFoV_multiColor(xMean, yMean,
                                                dx_list[i], dy_list[i], w=w)
        plt.savefig('/home/gcarla/workspace/toWindows/displacements/FilterJ/'
                    'multicolor_J_1_'+'%d' %(i))
        plt.close()


def main190327_plotAndSaveDisplacementsInImasCloseLoop_H(w=0.005):
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterH/Dither1/matchingTabs_list_H_dither1.pkl')
    ae = astrometricError_estimator.EstimateAstrometricError(tab)
    ae.createCubeOfStarsInfo()

    xMean = ae.getMeanPositionX()
    yMean = ae.getMeanPositionY()
    dx_list = []
    dy_list = []
    for i in range(len(tab)):
        dx, dy = ae.getDisplacementsFromMeanPositions(i)
        dx_list.append(dx)
        dy_list.append(dy)

    for i in range(len(dx_list)):
        # plt.figure()
        plotDisplacementsInLUCI1sFoV_multiColor(xMean, yMean,
                                                dx_list[i], dy_list[i], w=w)
        plt.savefig('/home/gcarla/workspace/toWindows/displacements/FilterH/'
                    'multicolor_H_1_'+'%d' %(i))
        plt.close()


def main190327_plotAndSaveDisplacementsInImasCloseLoop_K(w=0.005):
    tab = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/DataToRestore_forAnalysis/'
        'FilterKs/Dither1/matchingTabs_list_Ks_dither1.pkl')
    ae = astrometricError_estimator.EstimateAstrometricError(tab)
    ae.createCubeOfStarsInfo()

    xMean = ae.getMeanPositionX()
    yMean = ae.getMeanPositionY()
    dx_list = []
    dy_list = []
    for i in range(len(tab)):
        dx, dy = ae.getDisplacementsFromMeanPositions(i)
        dx_list.append(dx)
        dy_list.append(dy)

    for i in range(len(dx_list)):
        # plt.figure()
        plotDisplacementsInLUCI1sFoV_multiColor(xMean, yMean,
                                                dx_list[i], dy_list[i], w=w)
        plt.savefig('/home/gcarla/workspace/toWindows/displacements/FilterKs/'
                    'multicolor_Ks_1_'+'%d' %(i))
        plt.close()


def main190328_plotEPSFFwhm():

    def restoreAndFit(filterName, ditherList):
        root='/home/gcarla/workspace/20161019/DataToRestore_forAnalysis'
        res=[]
        for dither in ditherList:
            file= os.path.join(
                root, 'Filter%s/Dither%d/epsfModels_list_%s_dither%d.pkl' % (
                    filterName, dither, filterName, dither))
            epsfList = restoreObjectListFromFile(file)
            res1=[]
            for epsf in epsfList:
                res1.append(fitEPSFWithMoffat(epsf.data))
            res.append(res1)
        return np.array(res)

    resJ= restoreAndFit('J', [1, 2, 3, 5])
    fwhmJ= resJ[:, :, 2].flatten()
    resH= restoreAndFit('H', [1, 2, 3, 4])
    fwhmH= resH[:, :, 2].flatten()
    resK= restoreAndFit('Ks', [1, 2, 13])
    fwhmK= resK[:, :, 2].flatten()
    plt.figure()
    plt.plot(fwhmJ, label='filtro J')
    plt.plot(fwhmH, label='filtro H')
    plt.plot(fwhmK, label='filtro Ks')
    plt.ylabel('FWHM [arcsec]')
    plt.xlabel('# immagine')
    plt.legend()
    return resJ, resH, resK
