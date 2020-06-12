'''
Created on 10 mag 2020

@author: giuliacarla
'''

import numpy as np
from tesi import sandbox, select_stars, image_aligner
from ccdproc.combiner import Combiner


def main_alignImages():
    path1 = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/images_reduced/trimmed/NGC2419_K_dither_%d_trim.pkl'
    path2 = ' /Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/coordinates_for_alignment/trimmed/coords_NGC2419_K_dither%d.pkl'
    path3 = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/ARGOS/20161019/images_alignment/trimmed/NGC2419_K_dither%d_on_1.pkl)                        '
    n_init = 1
    n_fin = 13
    imas = [path1 % idx for idx in np.arange(n_init, n_fin + 1)]
    n_imas = len(imas)

    c_list = []
    for i in range(n_imas):
        se = select_stars.StarsSelector(imas[i])
        se.find_star_centroid(0)
        c1 = se.find_star_centroid
        se.find_star_centroid(1)
        c2 = se.find_star_centroid
        se.find_star_centroid(2)
        c3 = se.find_star_centroid
        coords = [c1, c2, c3]
        sandbox.saveObjectListToFile(coords, path2 % (i + 1))
        c_list.append(coords)

    ima_al_list = []
    for i in range(n_imas):
        al = image_aligner.ImageAligner(c_list[0], c_list[i + 1])
        ima_al = al.applyTransformationOnIma(imas[i + 1])
        sandbox.saveObjectListToFile(ima_al, path3 % (i + 1))
        ima_al_list.append(ima_al)

    ima_comb = Combiner(ima_al_list)
    ima_final = ima_comb.average_combine()
