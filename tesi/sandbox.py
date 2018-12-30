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
from astropy.units import amp
from astropy.table.table import Table
from photutils.datasets.make import make_gaussian_sources_image,\
    make_noise_image, apply_poisson_noise
from photutils import Background2D, MedianBackground
from astropy.stats import SigmaClip
from photutils.detection.findstars import IRAFStarFinder
from astropy.modeling import models, fitting
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_fwhm_to_sigma
from ccdproc.image_collection import ImageFileCollection
from tesi.image_fitter import ImageFitter
from tesi.image_creator import ImageCreator
from astropy import units as u
from hmac import new


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


def showNorm(ima, **kwargs):
    from astropy.visualization import imshow_norm, SqrtStretch
    from astropy.visualization.mpl_normalize import PercentileInterval
    plt.clf()
    if 'interval' not in kwargs:
        kwargs['interval']= PercentileInterval(99.7)
    if 'stretch' not in kwargs:
        kwargs['stretch']= SqrtStretch()
    if 'origin' not in kwargs:
        kwargs['origin']= 'lower'

    imshow_norm(ima, **kwargs)
#    imshow_norm(ima, origin='lower', interval=PercentileInterval(99.7),
#                stretch=SqrtStretch(), **kwargs)
    plt.colorbar()


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


def main181228():
    drFileName= '/home/gcarla/DataReduction20161019.pkl'
    from tesi.data_reduction import DataReduction
    dr = DataReduction.restoreFromFile(drFileName)
    dr.setFilterType('J')
    dr.setIntegrationTime(3.0)
    dr.setObjectName('NGC2419')
    return dr


def main181230():
    from tesi.detector import LuciDetector
    from ccdproc import Combiner

    dr= main181228()
    ima= dr.getScienceImage()
    darks= dr._darkIma
    flats= dr._flatIma
    skys= dr._skyIma
    sciences= dr._scienceIma

    # deviation, serve a quacosa?
    def _computeDeviation(ccd0, detector):
        cnew= ccdproc.create_deviation(
            ccd0,
            gain=detector.gainAdu2Electrons*u.electron/u.adu,
            readnoise=detector.ronInElectrons*u.electron)
        return cnew
    sciences0_new= _computeDeviation(sciences[0], LuciDetector())

    def _makeMasterDark(darks):
        darkCombiner= Combiner(darks)
        darkCombiner.sigma_clipping(low_thresh=3, high_thresh=3,
                                    func=np.ma.median, dev_func=np.ma.std)
        masterDark= darkCombiner.median_combine()
        masterDark.header['exptime']= darkCombiner.ccd_list[
            0].header['exptime']
        masterDark.header['DIT']= darkCombiner.ccd_list[0].header['DIT']
        # TODO: something else to be added to the masterDark.header?
        return masterDark

    def _adu2Electron(ccd):
        return ccdproc.gain_correct(ccd,
                                    LuciDetector().gainAdu2Electrons,
                                    u.electron/u.adu)

    def _makeMasterFlat(flats, masterDark):
        flatsDarkSubtracted=[]
        for flat in flats:
            flat= ccdproc.subtract_dark(
                flat, masterDark, exposure_time='DIT',
                exposure_unit=u.second,
                add_keyword={'calib': 'subtracted dark'})
            flatsDarkSubtracted.append(flat)

        flatCombiner= Combiner(flatsDarkSubtracted)
        flatCombiner.sigma_clipping(low_thresh=3, high_thresh=3,
                                    func=np.ma.median, dev_func=np.ma.std)

        def scalingFunc(arr):
            return 1./np.ma.average(arr)

        flatCombiner.scaling= scalingFunc
        masterFlat= flatCombiner.median_combine()
        masterFlat.header= flats[0].meta
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
