'''
Created on 16 feb 2019

@author: gcarla
'''

import numpy as np


class MatchTables():

    def __init__(self,
                 maxShift):
        self.maxShift = maxShift

    def match2Tables(self, t1, t2, coordType='photclass'):
        '''
        Assuming t1 and t2 contain a set of common stars shifted by
        less than maxShift pixel

        The function returns 2 tables containing only matching sources and
        sorted to have the some source ordering

        It assumes that the field is not crowded, so for a given source in t1
        there is at most one source in t2 within a radius maxShift

        Parameters
        ----------
        t1, t2: astropy tables to be matched.
        coordType: string to identify the tables setting.
            'irafclass' identifies tables obtained using the IRAFStarFinder
            class
            'photclass' identifies tables obtained using the PSFPhotometry
            class
            'wcs' identifies tables in which the stars coordinates are in WCS
            (World Coordinate Sysyem)
        '''
        pass
        t1Out = t1.copy()
        t1Out.remove_rows(range(len(t1Out)))
        t2Out = t2.copy()
        t2Out.remove_rows(range(len(t2Out)))
        if coordType == 'irafclass':
            self.coord1 = np.vstack([np.array(t1['xcentroid']),
                                     np.array(t1['ycentroid'])]).T
            self.coord2 = np.vstack([np.array(t2['xcentroid']),
                                     np.array(t2['ycentroid'])]).T
        elif coordType == 'photclass':
            self.coord1 = np.vstack([np.array(t1['x_fit']),
                                     np.array(t1['y_fit'])]).T
            self.coord2 = np.vstack([np.array(t2['x_fit']),
                                     np.array(t2['y_fit'])]).T
        elif coordType == 'wcs':
            self.coord1 = np.vstack([np.array(t1['ra']),
                                     np.array(t1['dec'])]).T
            self.coord2 = np.vstack([np.array(t2['ra']),
                                     np.array(t2['dec'])]).T

        for i in range(self.coord2.shape[0]):
            closeToSource2 = np.linalg.norm(self.coord1 - self.coord2[i, :],
                                            axis=1) < self.maxShift
            if np.count_nonzero(closeToSource2) > 1:
                raise Exception('Found more than 1 matching source for '
                                'source at %s' % self.coord2[i])
            elif np.count_nonzero(closeToSource2) == 1:
                c1Idx = np.argwhere(closeToSource2)[0][0]
                t2Out.add_row(t2[i])
                t1Out.add_row(t1[c1Idx])
        return t1Out, t2Out

    def matchListOfTables(self, tabsList):
        refTab = self.match2Tables(tabsList[0], tabsList[1])[0]
        for tab in tabsList[2:]:
            #  print('Matching tab %g with reference tab' % tab)
            _, refTab = self.match2Tables(tab, refTab)
        matchingFitTabs = []
        for tab in tabsList:
            #  print('Matching tab %g with reference tab' %fitTab)
            tab1, _ = self.match2Tables(tab, refTab)
            matchingFitTabs.append(tab1)
        return matchingFitTabs

#
# def matchStarsTablesListWithRefTab(tabsList, refTab, max_shift):
#     match = match_astropy_tables.MatchTables(max_shift)
#     #refTab = match.match2TablesPhotometry(tabsList[0], tabsList[1])[0]
#     for fitTab in tabsList:
#         _, refTab = match.match2TablesPhotometry(fitTab, refTab)
#     matchingFitTabs = []
#     for fitTab in tabsList:
#         tab1, _ = match.match2TablesPhotometry(fitTab, refTab)
#         matchingFitTabs.append(tab1)
#     return matchingFitTabs
