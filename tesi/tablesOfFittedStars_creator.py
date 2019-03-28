'''
Created on 05 mar 2019

@author: gcarla
'''
from tesi import ePSF_builder, image_fitter, match_astropy_tables


class createTablesListOfFittedStars():
    '''
    This class creates a tables list from stars fitting
    of a list of images. Every table will contain same stars.
    Fitting is obtained from 
     - ePSF building
     - LSQ fit using the DAOPHOT algorithm with ePSF as PSF model
    '''

    def __init__(self, imasList):
        self.imas = imasList

    def _buildEPSFOnSingleImage(self, ima, threshold, fwhm, minSep,
                                sharplo, sharphi, roundlo,
                                roundhi, peakmax, size):
        self.builder = ePSF_builder.ePSFBuilder(ima, threshold, fwhm, minSep,
                                                sharplo, sharphi, roundlo,
                                                roundhi, peakmax, size)
        self.builder.removeBackground()
        self.builder.extractStars()
        self.builder.selectGoodStars()
        self.builder.buildEPSF()

    def _fitStarsWithePSFModelOnSingleImage(self, ima, threshold, fwhm,
                                            min_sep, sharplo, sharphi,
                                            roundlo, roundhi, peakmax,
                                            psfModel, fitshape, aperture_rad):
        ima_fit = image_fitter.ImageFitter(threshold, fwhm, min_sep,
                                           sharplo, sharphi,
                                           roundlo, roundhi,
                                           peakmax)
        ima_fit.fitStarsWithBasicPhotometry(ima,
                                            psfModel,
                                            fitshape,
                                            aperture_rad)
        fitTab = ima_fit.getFitTable()
        return fitTab

    def _matchStarsTablesList(self, tabsList):
        match = match_astropy_tables.MatchTables(2)
        refTab = match.match2TablesPhotometry(tabsList[0], tabsList[1])[0]
        for fitTab in tabsList[2:]:
            _, refTab = match.match2TablesPhotometry(fitTab, refTab)

        matchingFitTabs = []
        for fitTab in tabsList:
            tab1, _ = match.match2TablesPhotometry(fitTab, refTab)
            matchingFitTabs.append(tab1)
        return matchingFitTabs

    def makeEPSFModelsList(self, threshold=8e03, fwhm=3., minSep=10.,
                           sharplo=0.1, sharphi=2.0, roundlo=-1.0,
                           roundhi=1.0, peakmax=5e04,
                           size=50):
        self.epsfList = []
        for ima in self.imas:
            self._buildEPSFOnSingleImage(ima, threshold, fwhm, minSep,
                                         sharplo, sharphi, roundlo,
                                         roundhi, peakmax, size)
            psfModel = self.builder.getEPSFModel()
            self.epsfList.append(psfModel)

    def fitStarsPositionOnImasList(self, threshold=1e03, fwhm=3.,
                                   min_sep=3., sharplo=0.1,
                                   sharphi=2.0, roundlo=-1.0,
                                   roundhi=1.0, peakmax=5e04,
                                   fitshape=(21, 21), aperture_rad=45):
        self.starsTabs = []
        for i in range(len(self.imas)):
            tab = self._fitStarsWithePSFModelOnSingleImage(
                self.imas[i],
                threshold, fwhm, min_sep,
                sharplo, sharphi, roundlo,
                roundhi, peakmax, self.epsfList[i],
                fitshape, aperture_rad)
            self.starsTabs.append(tab)

    def getListOfMatchingTabs(self):
        self.matchingTabs = self._matchStarsTablesList(self.starsTabs)
        #print('N stars: %d' %len(self.matchingTabs[0]))
        return self.matchingTabs
