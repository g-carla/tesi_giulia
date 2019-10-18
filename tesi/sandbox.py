'''
Created on 27 set 2018

@author: gcarla
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tesi import image_filter, image_cleaner, image_fitter
from photutils.detection.findstars import IRAFStarFinder
from astropy.nddata.utils import Cutout2D
from tesi import ePSF_builder
from tesi import match_astropy_tables
from tesi import astrometric_error_estimator


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


def plot3D(xShape, yShape, data, cbarMin=None, cbarMax=None, **kwargs):
    #     from matplotlib.ticker import LinearLocator, FormatStrFormatter
    #    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    y, x = np.mgrid[0:yShape, 0:xShape]
    surf = ax.plot_surface(X=x, Y=y, Z=data, cmap='viridis', **kwargs)

#    # Customize the z axis.
#     ax.set_zlim(-1.01, 1.01)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if (cbarMin and cbarMax) is not None:
        surf.set_clim(cbarMin, cbarMax)


def saveObjectListToFile(objList, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(objList, handle, protocol=pickle.HIGHEST_PROTOCOL)


def restoreObjectListFromFile(filename):
    with open(filename, 'rb') as handle:
        objList= pickle.load(handle)
    return objList


def getDarksForReduction():
    imFil = image_filter.ImageFilter('/home/gcarla/tera1/201610/data/20161019')
    darkImas, darkNames = imFil.getLists('DARK', 3.0)
    return darkImas, darkNames


def getFlatsForReduction():
    imFil = image_filter.ImageFilter('/home/gcarla/tera1/201610/data/20161019')
    flatImasJ, flatNamesJ = imFil.getLists('FLAT', 3.0, FILTER='J')
    flatImasH, flatNamesH = imFil.getLists('FLATFIELD', 3.0, FILTER='H')
    flatImasK, flatNamesK = imFil.getLists('FLATFIELD', 3.0, FILTER='Ks')
    # WRONG HEADER
    del flatImasJ[11:28]
    del flatNamesJ[11:28]
    ##
    return flatImasJ, flatNamesJ, flatImasH, flatNamesH, flatImasK, flatNamesK


def getSkiesForReduction():
    imFil = image_filter.ImageFilter('/home/gcarla/tera1/201610/data/20161019')
    skyImasJ, skyNamesJ = imFil.getLists('SKY', 3.0, FILTER='J',
                                         OBJECT='NGC2419')
    skyImasH, skyNamesH = imFil.getLists('SKY', 3.0, FILTER='H',
                                         OBJECT='NGC2419')
    skyImasK, skyNamesK = imFil.getLists('SKY', 3.0, FILTER='Ks',
                                         OBJECT='NGC2419')
    del skyImasJ[35:]
    del skyNamesJ[35:]
    return skyImasJ, skyNamesJ, skyImasH, skyNamesH, skyImasK, skyNamesK


def getScisForReduction():
    '''
    ATTENTION: all dithers in sciLists. For reduction: check the log file
    and split sciImas and sciNames in N lists (N = number of dithers).
    Example: J_imas_dither1 = sciImasJ[0:9]
    '''

    imFil = image_filter.ImageFilter('/home/gcarla/tera1/201610/data/20161019')
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


def plotAstrometricErrorOnLuciField(starsTabs, area=40, pathStr=None):
    ae = astrometric_error_estimator.EstimateAstrometricError(starsTabs)
    ae.plotAstroErrorOntheField(area=area)
    if pathStr is not None:
        plt.savefig(pathStr)


def plotStarsShiftFromMeanPosition(starsTabs, color, scale=0.002,
                                   pathStr=None):
    ae = astrometric_error_estimator.EstimateAstrometricError(starsTabs)

    for i in range(len(starsTabs)):
        dx, dy = ae.getDisplacementsFromMeanPositions(i)
        ae.plotDisplacements(dx, dy, color=color, scale=scale)
        if pathStr is not None:
            plt.savefig(pathStr + '%d' %(i+1))
            # plt.close()


def plotDifferentialTiltJitterError(starsTabs, NGSCoords, n,
                                    leg='yes', pathStr=None):
    ae = astrometric_error_estimator.EstimateDifferentialTiltJitter(starsTabs,
                                                                    NGSCoords,
                                                                    n=n)
    ae.plotDTJError(leg=leg)
    if pathStr is not None:
        plt.savefig(pathStr)


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
