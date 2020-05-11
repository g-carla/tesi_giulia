'''
@author: giuliacarla
'''

from tesi import image_cleaner
from tesi import sandbox


def reduceFilterJDither1():
    darks = sandbox.restoreObjectListFromFile(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/raw/'
        'dark_imas_list.pkl')
    flatsJ = sandbox.restoreObjectListFromFile(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/'
        'raw/flat_J_imas_list.pkl')
    skiesJ = sandbox.restoreObjectListFromFile(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/'
        'raw/sky_J_imas_list.pkl')
    scisJ_dith1 = sandbox.restoreObjectListFromFile(
        '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/raw/'
        'NGC2419_J_imas_list_dither1.pkl')
    ima_cle = image_cleaner.ImageCleaner(scisJ_dith1, darks, flatsJ, skiesJ)
    mastersci = ima_cle.getScienceImage()
    return mastersci
