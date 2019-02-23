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
from tesi.image_fitter import ImageFitter
from tesi.image_creator import ImageCreator
from astropy import units as u
from astropy.stats.funcs import gaussian_sigma_to_fwhm
from tesi import image_creator, image_fitter, data_reduction, image_aligner,\
    astrometricError_estimator, ePSF_builder, match_astropy_tables,\
    theoretical_astrometricError_test
from ccdproc.ccddata import CCDData
from skimage.transform._warps import warp
from matplotlib.pyplot import xlabel, ylabel, xticks, yticks, plot
from cmath import pi


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


def show(ima, **kwargs):
    plt.clf()
    plt.imshow(ima, origin='lower', cmap='gray', **kwargs)
    plt.colorbar()


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

    fig.colorbar(im)


def showImaLuci(ima):
    showNorm(ima, cmap='gray', extent=[-120, 120, -120, 120])
    xlabel('arcsec', size=12)
    ylabel('arcsec', size=12)
    xticks(size=11)
    yticks(size=11)


def _showNormOld(ima, **kwargs):
    from astropy.visualization import simple_norm
    plt.clf()
    norm= simple_norm(ima, 'linear', percent=99.5)
    plt.imshow(ima, origin='lower', norm=norm)
    plt.colorbar()


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


def removebkg(ima):
    mask= make_source_mask(ima, snr=2, npixels=5, dilate_size=7)
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
    fname = '/home/gcarla/workspace/20161019/reduction/FilterJ/IndividualFrames_Dither1/NGC2419_20161019_luci1_List.pkl'
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
        '/home/gcarla/workspace/20161019/reduction/FilterJ/IndividualFrames_Dither1/NGC2419_20161019_luci1_ListFITS.pkl')
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
        '/home/gcarla/workspace/20161019/plot/main190216/matchingTablesList.pkl')
    imas = restoreObjectListFromFile(
        '/home/gcarla/workspace/20161019/reduction/FilterJ/IndividualFrames_Dither1/NGC2419_20161019_luci1_ListFITS.pkl')
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
