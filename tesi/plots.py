'''
@author: giuliacarla
'''

import numpy as np
import matplotlib.pyplot as plt
from tesi import astrometric_error_estimator
from astropy.visualization.interval import PercentileInterval


def plot3D(data, xShape, yShape, cbarMin=None, cbarMax=None, **kwargs):
    #     from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d import Axes3D

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


def showNorm(imaOrCcd, **kwargs):
    from astropy.visualization import imshow_norm, SqrtStretch
    from astropy.visualization.mpl_normalize import PercentileInterval
    from astropy.nddata import CCDData

    plt.clf()
    fig = plt.gcf()
    if isinstance(imaOrCcd, CCDData):
        arr = imaOrCcd.data
        wcs = imaOrCcd.wcs
        if wcs is None:
            ax = plt.subplot()
        else:
            ax = plt.subplot(projection=wcs)
            ax.coords.grid(True, color='white', ls='solid')
    else:
        arr = imaOrCcd
        ax = plt.subplot()
    if 'interval' not in kwargs:
        kwargs['interval'] = PercentileInterval(99.7)
    if 'stretch' not in kwargs:
        kwargs['stretch'] = SqrtStretch()
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'

    im, _ = imshow_norm(arr, ax=ax, **kwargs)

    cb = fig.colorbar(im)
    cb.ax.tick_params(labelsize=11)


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
            plt.savefig(pathStr + '%d' % (i + 1))
            # plt.close()


def plotDifferentialTiltJitterError(starsTabs, NGSCoords, n,
                                    leg='yes', pathStr=None):
    ae = astrometric_error_estimator.EstimateDifferentialTiltJitter(starsTabs,
                                                                    NGSCoords,
                                                                    n=n)
    ae.plotDTJError(leg=leg)
    if pathStr is not None:
        plt.savefig(pathStr)


def showStarsFromIRAFTable(ima, table, color,
                           perc_interval=98, aperture_radius=7):
    from photutils.aperture.circle import CircularAperture

    positions = (table['xcentroid'], table['ycentroid'])
    apertures = CircularAperture(positions, r=aperture_radius)
    showNorm(ima, interval=PercentileInterval(perc_interval))
    apertures.plot(color=color)


def showStarsFromPhotometryTable(ima, table, color,
                                 perc_interval=98, aperture_radius=7):
    from photutils.aperture.circle import CircularAperture

    positions = (table['x_fit'], table['y_fit'])
    apertures = CircularAperture(positions, r=aperture_radius)
    showNorm(ima, interval=PercentileInterval(perc_interval))
    apertures.plot(color=color)
