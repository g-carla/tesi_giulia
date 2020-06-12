'''
Created on 7 giu 2020

@author: giuliacarla
'''

import numpy as np
import matplotlib.pyplot as plt
from tesi.plots import showNorm
from astropy.visualization import PercentileInterval
from astropy.nddata.utils import Cutout2D
from tesi import plots
from photutils.centroids.core import centroid_2dg
from photutils.detection.core import find_peaks


RADIUS = 30


class StarsSelector():

    def __init__(self, image):
        #         plt.ion()
        self._ima = image
        self.fig = plt.figure()
        showNorm(image, interval=PercentileInterval(99.5))
        plt.title("Left click to select a star. End with right click.")
        self.ax = self.fig.gca()
        cid1 = self.fig.canvas.mpl_connect("button_press_event",
                                           self._press)
        cid2 = self.fig.canvas.mpl_connect("button_release_event",
                                           self._release)
        self.goon = True
        self.circles = []
        plt.show()
        self._start()

    def _start(self):
        while self.goon:
            plt.pause(0.10)

    def _press(self, event):
        if event.button == 1:
            self.xc = event.xdata
            self.yc = event.ydata

    def _release(self, event):
        if event.button == 1:
            circle = plt.Circle(
                (self.xc, self.yc), RADIUS, color="r", fill=False)
            self.ax.add_artist(circle)
            self.circles.append([self.xc, self.yc])
            plt.draw()
        elif event.button == 3:
            self.goon = False
        self.n_stars = len(self.circles)

    def find_star_centroid(self, n_star, centroid_func=centroid_2dg):
        raw_center = np.array(self.circles[n_star])
        self._cutImage(raw_center)
        print('Choose threshold:')
        threshold = float(input())
        print('Choose box size:')
        box_size = int(input())
        tab = find_peaks(
            data=self.cut.data, threshold=threshold,
            box_size=box_size,
            centroid_func=centroid_func)

        self.star_centroid_cut_ima = np.array([
            tab['x_centroid'][0], tab['y_centroid'][0]
        ])
        self.star_centroid = self.star_centroid_cut_ima + (
            raw_center - RADIUS / 2)

    def _cutImage(self, circle_center):
        self.cut = Cutout2D(self._ima, circle_center, RADIUS)
        plt.figure()
        plots.showNorm(self.cut.data)
        plt.title('Cut image')
