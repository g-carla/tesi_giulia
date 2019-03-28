'''
Created on 16 feb 2019

@author: gcarla
'''

import numpy as np


class MatchTables():
    '''
    This class...
    '''

    def __init__(self,
                 maxShift):
        self.maxShift = maxShift

    def match2Tables(self, t1, t2):
        '''
        Assuming t1 and t2 are tables returned from IRAFStarFinder
        Assuming t1 and t2 contains a set of common stars shifted by
        less than maxShift pixel

        The function returns 2 tables containing only matching sources and
        sorted to have the some source ordering

        It assumes that the field is not crowded, so for a given source in t1
        there is at most one source in t2 within a radius maxShift
        '''
        pass
        t1Out= t1.copy()
        t1Out.remove_rows(range(len(t1Out)))
        t2Out= t2.copy()
        t2Out.remove_rows(range(len(t2Out)))
        self.coord1= np.vstack([np.array(t1['xcentroid']),
                                np.array(t1['ycentroid'])]).T
        self.coord2= np.vstack([np.array(t2['xcentroid']),
                                np.array(t2['ycentroid'])]).T
        for i in range(self.coord2.shape[0]):
            closeToSource2= np.linalg.norm(self.coord1-self.coord2[i, :],
                                           axis=1) < self.maxShift
            if np.count_nonzero(closeToSource2) > 1:
                raise Exception('Found more than 1 matching source for '
                                'source at %s' % self.coord2[i])
            elif np.count_nonzero(closeToSource2) == 1:
                c1Idx= np.argwhere(closeToSource2)[0][0]
                t2Out.add_row(t2[i])
                t1Out.add_row(t1[c1Idx])
        return t1Out, t2Out

    def match2TablesPhotometry(self, t1, t2):
        '''
        Assuming t1 and t2 are tables returned from the PSFPhotometry class
        Assuming t1 and t2 contains a set of common stars shifted by
        less than maxShift pixel

        The function returns 2 tables containing only matching sources and
        sorted to have the some source ordering

        It assumes that the field is not crowded, so for a given source in table1
        there is at most one source in t2 within a radius maxShift
        '''
        pass
        t1Out= t1.copy()
        t1Out.remove_rows(range(len(t1Out)))
        t2Out= t2.copy()
        t2Out.remove_rows(range(len(t2Out)))
        self.c1= np.vstack([np.array(t1['x_fit']), np.array(t1['y_fit'])]).T
        self.c2= np.vstack([np.array(t2['x_fit']), np.array(t2['y_fit'])]).T
        for i in range(self.c2.shape[0]):
            closeToSource2= np.linalg.norm(self.c1-self.c2[i, :],
                                           axis=1) < self.maxShift
            if np.count_nonzero(closeToSource2) > 1:
                raise Exception('Found more than 1 matching source for '
                                'source at %s' % self.c2[i])
            elif np.count_nonzero(closeToSource2) == 1:
                c1Idx= np.argwhere(closeToSource2)[0][0]
                t2Out.add_row(t2[i])
                t1Out.add_row(t1[c1Idx])
        return t1Out, t2Out

    def match2TablesWorldCoords(self, t1, t2):
        '''
        Assuming t1 and t2 are tables returned from the PSFPhotometry class
        Assuming t1 and t2 contains a set of common stars shifted by
        less than maxShift pixel

        The function returns 2 tables containing only matching sources and
        sorted to have the some source ordering

        It assumes that the field is not crowded, so for a given source in table1
        there is at most one source in t2 within a radius maxShift
        '''
        pass
        t1Out= t1.copy()
        t1Out.remove_rows(range(len(t1Out)))
        t2Out= t2.copy()
        t2Out.remove_rows(range(len(t2Out)))
        self.c1= np.vstack([np.array(t1['ra']), np.array(t1['dec'])]).T
        self.c2= np.vstack([np.array(t2['ra']), np.array(t2['dec'])]).T
        for i in range(self.c2.shape[0]):
            closeToSource2= np.linalg.norm(self.c1-self.c2[i, :],
                                           axis=1) < self.maxShift
            if np.count_nonzero(closeToSource2) > 1:
                raise Exception('Found more than 1 matching source for '
                                'source at %s' % self.c2[i])
            elif np.count_nonzero(closeToSource2) == 1:
                c1Idx= np.argwhere(closeToSource2)[0][0]
                t2Out.add_row(t2[i])
                t1Out.add_row(t1[c1Idx])
        return t1Out, t2Out
