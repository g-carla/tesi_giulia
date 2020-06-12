'''
@author: giuliacarla
'''
from photutils.background.background_2d import Background2D


def remove_2D_background(image, mask=None, box_size=50):
    bg = Background2D(data=image, mask=mask, box_size=box_size)
    return image - bg.background
