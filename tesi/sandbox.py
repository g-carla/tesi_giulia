'''
Created on 27 set 2018

@author: gcarla
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tesi import image_filter, image_cleaner, image_fitter, plots
from photutils.detection.findstars import IRAFStarFinder, DAOStarFinder
from astropy.nddata.utils import Cutout2D
from tesi import ePSF_builder
from tesi import match_astropy_tables
from tesi import astrometric_error_estimator
from photutils.centroids.core import centroid_2dg
from photutils.detection.core import find_peaks


def saveObjectListToFile(objList, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(objList, handle, protocol=pickle.HIGHEST_PROTOCOL)


def restoreObjectListFromFile(filename):
    with open(filename, 'rb') as handle:
        objList = pickle.load(handle)
    return objList


def getDarksForReduction(dir_path='/home/gcarla/tera1/201610/data/20161019'):
    imFil = image_filter.ImageFilter(dir_path)
    darkImas, darkNames = imFil.getLists('DARK', 3.0)
    return darkImas, darkNames


def getFlatsForReduction(dir_path='/home/gcarla/tera1/201610/data/20161019'):
    imFil = image_filter.ImageFilter(dir_path)
    flatImasJ, flatNamesJ = imFil.getLists('FLAT', 3.0, FILTER='J')
    flatImasH, flatNamesH = imFil.getLists('FLATFIELD', 3.0, FILTER='H')
    flatImasK, flatNamesK = imFil.getLists('FLATFIELD', 3.0, FILTER='Ks')
    # WRONG HEADER
    del flatImasJ[11:28]
    del flatNamesJ[11:28]
    ##
    return flatImasJ, flatNamesJ, flatImasH, flatNamesH, flatImasK, flatNamesK


def getSkiesForReduction(dir_path='/home/gcarla/tera1/201610/data/20161019'):
    imFil = image_filter.ImageFilter(dir_path)
    skyImasJ, skyNamesJ = imFil.getLists('SKY', 3.0, FILTER='J',
                                         OBJECT='NGC2419')
    skyImasH, skyNamesH = imFil.getLists('SKY', 3.0, FILTER='H',
                                         OBJECT='NGC2419')
    skyImasK, skyNamesK = imFil.getLists('SKY', 3.0, FILTER='Ks',
                                         OBJECT='NGC2419')
    del skyImasJ[35:]
    del skyNamesJ[35:]
    return skyImasJ, skyNamesJ, skyImasH, skyNamesH, skyImasK, skyNamesK


def getScisForReduction(dir_path='/home/gcarla/tera1/201610/data/20161019'):
    '''
    ATTENTION: all dithers in sciLists. For reduction: check the log file
    and split sciImas and sciNames in N lists (N = number of dithers).
    Example: J_imas_dither1 = sciImasJ[0:9]
    '''

    imFil = image_filter.ImageFilter(dir_path)
    sciImasJ, sciNamesJ = imFil.getLists('SCIENCE', 3.0, FILTER='J',
                                         OBJECT='NGC2419')
    sciImasH, sciNamesH = imFil.getLists('SCIENCE', 3.0, FILTER='H',
                                         OBJECT='NGC2419')
    sciImasK, sciNamesK = imFil.getLists('SCIENCE', 3.0, FILTER='Ks',
                                         OBJECT='NGC2419')
    # NOT GOOD IMAGES
    del sciImasH[56:]
    del sciNamesH[56:]
    ##
    return sciImasJ, sciNamesJ, sciImasH, sciNamesH, sciImasK, sciNamesK


def getNGC2419JDithers(sciJImas, sciJFilenames):
    ima_dith1 = sciJImas[:9]
    ima_dith2 = sciJImas[9:18]
    ima_dith3 = sciJImas[18:27]
    ima_dith4 = sciJImas[27:36]
    ima_dith5 = sciJImas[36:45]
    ima_dith6 = sciJImas[45:54]
    ima_dith7 = sciJImas[54:63]
    ima_dith8 = sciJImas[63:72]

    names_dith1 = sciJFilenames[:9]
    names_dith2 = sciJFilenames[9:18]
    names_dith3 = sciJFilenames[18:27]
    names_dith4 = sciJFilenames[27:36]
    names_dith5 = sciJFilenames[36:45]
    names_dith6 = sciJFilenames[45:54]
    names_dith7 = sciJFilenames[54:63]
    names_dith8 = sciJFilenames[63:72]

    return ima_dith1, ima_dith2, ima_dith3, ima_dith4, ima_dith5, ima_dith6, \
        ima_dith7, ima_dith8, \
        names_dith1, names_dith2, names_dith3, names_dith4, names_dith5, \
        names_dith6, names_dith7, names_dith8


def getNGC2419HDithers(sciHImas, sciHFilenames):
    ima_dith1 = sciHImas[:14]
    ima_dith2 = sciHImas[14:28]
    ima_dith3 = sciHImas[28:42]
    ima_dith4 = sciHImas[42:56]

    names_dith1 = sciHFilenames[:14]
    names_dith2 = sciHFilenames[14:28]
    names_dith3 = sciHFilenames[28:42]
    names_dith4 = sciHFilenames[42:56]

    return ima_dith1, ima_dith2, ima_dith3, ima_dith4, \
        names_dith1, names_dith2, names_dith3, names_dith4


def getNGC2419KDithers(sciKImas, sciKFilenames):
    ima_dith1 = sciKImas[:11]
    ima_dith2 = sciKImas[11:22]
    ima_dith3 = sciKImas[22:33]
    ima_dith4 = sciKImas[33:44]
    ima_dith5 = sciKImas[44:55]
    ima_dith6 = sciKImas[55:66]
    ima_dith7 = sciKImas[66:77]
    ima_dith8 = sciKImas[77:88]
    ima_dith9 = sciKImas[88:99]
    ima_dith10 = sciKImas[99:110]
    ima_dith11 = sciKImas[121:132]
    ima_dith12 = sciKImas[132:143]
    ima_dith13 = sciKImas[143:154]

    names_dith1 = sciKFilenames[:11]
    names_dith2 = sciKFilenames[11:22]
    names_dith3 = sciKFilenames[22:33]
    names_dith4 = sciKFilenames[33:44]
    names_dith5 = sciKFilenames[44:55]
    names_dith6 = sciKFilenames[55:66]
    names_dith7 = sciKFilenames[66:77]
    names_dith8 = sciKFilenames[77:88]
    names_dith9 = sciKFilenames[88:99]
    names_dith10 = sciKFilenames[99:110]
    names_dith11 = sciKFilenames[121:132]
    names_dith12 = sciKFilenames[132:143]
    names_dith13 = sciKFilenames[143:154]

# Ultimo dither sfuocato...

    return ima_dith1, ima_dith2, ima_dith3, ima_dith4, ima_dith5, ima_dith6, \
        ima_dith7, ima_dith8, ima_dith9, ima_dith10, ima_dith11, ima_dith12, \
        ima_dith13, \
        names_dith1, names_dith2, names_dith3, names_dith4, names_dith5, \
        names_dith6, names_dith7, names_dith8, names_dith9, names_dith10, \
        names_dith11, names_dith12, names_dith13


def reduceListOfNGC2419Imas(imasRaw, darks, flats, skies, masterDark=None,
                            masterFlat=None, masterSky=None):
    imasReduced = []
    imCle = image_cleaner.ImageCleaner(darkImas=darks, flatImas=flats,
                                       skyImas=skies, sciImas=None)
    if masterDark is not None:
        imCle.setMasterDark(masterDark)
    if masterFlat is not None:
        imCle.setMasterFlat(masterFlat)
    if masterSky is not None:
        imCle.setMasterSky(masterSky)
    for ima in imasRaw:
        imCle.setScienceImage(ima)
        imaNew = imCle.getScienceImage()
        imasReduced.append(imaNew)
    return imasReduced


def getEPSFsListFromImasList(imaList, threshold=8e03, fwhm=3., minSep=10.,
                             sharplo=0.1, sharphi=2.0, roundlo=-1.0,
                             roundhi=1.0, peakmax=5e04, size=50):
    epsfList = []
    for ima in imaList:
        builder = ePSF_builder.epsfBuilder(ima, threshold, fwhm, minSep,
                                           sharplo, sharphi, roundlo, roundhi,
                                           peakmax, size)
        builder.removeBackground()
        builder.buildEPSF()
        epsf = builder.getEPSFModel()
        epsfList.append(epsf)
    return epsfList


def getListOfStarsTabsFromImasList(imaList, epsfList, thresh=1e03, fwhm=3.,
                                   min_sep=3., sharplo=0.1, sharphi=2.0,
                                   roundlo=-1.0, roundhi=1.0,
                                   fitshape=(21, 21), aperture_rad=45,
                                   peakmax=5e04):
    imFit = image_fitter.StarsFitter(image=None, thresholdInPhot=thresh,
                                     fwhm=fwhm, min_separation=min_sep,
                                     sharplo=sharplo, sharphi=sharphi,
                                     roundlo=roundlo, roundhi=roundhi,
                                     fitshape=fitshape,
                                     apertureRadius=aperture_rad,
                                     peakmax=peakmax)
    tabsList = []
    for i in range(len(imaList)):
        imFit.setImage(imaList[i])
        imFit.fitStars(epsfList[i])
        tabsList.append(imFit.getFitTable())
    return tabsList


def matchListOfStarsTabs(tabsList, maxShift=2):
    matchTab = match_astropy_tables.MatchTables(maxShift=maxShift)
    refTab = matchTab.match2Tables(tabsList[0], tabsList[1])[0]
    for tab in tabsList[2:]:
        _, refTab = matchTab.match2Tables(tab, refTab)

    matchingTabs = []
    for tab in tabsList:
        tab1, _ = matchTab.match2Tables(tab, refTab)
        matchingTabs.append(tab1)
    return matchingTabs


class IRAFStarFinderExcludingMaskedPixel(IRAFStarFinder):

    def __init__(self, *args, **kwargs):
        super(IRAFStarFinderExcludingMaskedPixel, self).__init__(
            *args, **kwargs)

    def _cutDataOnABoxAroundStar(self, ima, xc, yc):
        cut = Cutout2D(ima, (xc, yc), 21)
        return cut.data

    def find_stars(self, data, mask=None):
        self.table = super(IRAFStarFinderExcludingMaskedPixel,
                           self).find_stars(data, mask)

#     if mask is not None:
#     else: return self.table
        if data.mask.any():
            self.tableCopy = self.table.copy()
            self.tableCopy.remove_rows(range(len(self.tableCopy)))

            for i in range(len(self.table) - 1):
                imaCut = self._cutDataOnABoxAroundStar(
                    data,
                    self.table[i]['xcentroid'],
                    self.table[i]['ycentroid'])
                if imaCut.mask.any() is False:
                    self.tableCopy.add_row(self.table[i])
            return self.tableCopy
        else:
            return self.table


class DAOStarFinderExcludingMaskedPixel(DAOStarFinder):

    def __init__(self, *args, **kwargs):
        super(DAOStarFinderExcludingMaskedPixel, self).__init__(
            *args, **kwargs)

    def _cutDataOnABoxAroundStar(self, ima, xc, yc):
        cut = Cutout2D(ima, (xc, yc), 21)
        return cut.data

    def find_stars(self, data, mask=None):
        self.table = super(DAOStarFinderExcludingMaskedPixel, self).find_stars(
            data, mask)

#     if mask is not None:
#     else: return self.table
        if data.mask.any():
            self.tableCopy = self.table.copy()
            self.tableCopy.remove_rows(range(len(self.tableCopy)))

            for i in range(len(self.table) - 1):
                imaCut = self._cutDataOnABoxAroundStar(
                    data,
                    self.table[i]['xcentroid'],
                    self.table[i]['ycentroid'])
                if imaCut.mask.any() is False:
                    self.tableCopy.add_row(self.table[i])
            return self.tableCopy
        else:
            return self.table


def convertNanPixelsInMedianOfImage(ima, nanIndices):
    yy = nanIndices[:, 0]
    xx = nanIndices[:, 1]
    median = np.median(ima)
    ima[yy, xx] = median
    return ima


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
    import astropy.units as u
    from astropy import wcs
    from astropy.coordinates import SkyCoord, FK5

    sciences = dr._scienceIma
    ccd0 = sciences[0]
    hdr0 = ccd0.header
    ccd0wcs = wcs.WCS(hdr0)
    pxs = np.array([[0, 0], [1024, 1024], [512, 1024]], np.float)
    ccd0wcs.all_pix2world(pxs, 1)

    px = np.arange(ccd0.shape[1])
    py = np.arange(ccd0.shape[0])
    wx, wy = ccd0wcs.all_pix2world(px, py, 1)

    if hdr0['OBJRADEC'] == 'FK5':
        frameType = FK5()
    c = SkyCoord(ccd0.header['OBJRA'], ccd0.header['OBJDEC'],
                 frame=frameType, unit=(u.hourangle, u.deg))

    # AO guide star. Find it in image
    aoStarCoordW = SkyCoord(ccd0.header['AORA'], ccd0.header['AODEC'],
                            frame=frameType, unit=(u.hourangle, u.deg))
    aoStarCoordPx = ccd0wcs.world_to_pixel(aoStarCoordW)
