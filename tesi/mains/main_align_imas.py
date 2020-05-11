'''
Created on 10 mag 2020

@author: giuliacarla
'''

from tesi import sandbox, match_astropy_tables, image_aligner


def alignKDither1AndDither2():
    tab1 = sandbox.restoreObjectListFromFile(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/'
        'fit_tables/fit_tables_K_dither_1.pkl')
    tab2 = sandbox.restoreObjectListFromFile(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/'
        'fit_tables/fit_tables_K_dither_2.pkl')
    ima1 = sandbox.restoreObjectListFromFile(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/'
        'reduced/NGC2419_K_dither_1.pkl')
    ima2 = sandbox.restoreObjectListFromFile(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/'
        'reduced/NGC2419_K_dither_2.pkl')

    mm = match_astropy_tables.MatchTables(maxShift=10)
    tab1New, tab2New = mm.match2Tables(tab1, tab2)

    ima_al = image_aligner.ImageAligner(tab1New, tab2New)
    ima_al.findTransformationMatrixWithDAOPHOTTable()
    ima2OnIma1 = ima_al.applyTransformationOnIma(ima2)

    return ima1, ima2, ima2OnIma1
