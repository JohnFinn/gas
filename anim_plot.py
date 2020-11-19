#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import math


class GraphAnimation:

    def __init__(self, label1, label2):
        self.hl1, = plt.plot([], [], label=label1)
        self.hl2, = plt.plot([], [], label=label2)
        plt.legend()
        self.left, self.right = self.bottom, self.top = (-1, 1)

    def update_xylim(self, x, y):
        self.left   = min(min(x) - 0.1, self.left)
        self.right  = max(max(x) + 0.1, self.right)
        self.bottom = min(min(y) - 0.1, self.bottom)
        self.top    = max(max(y) + 0.1, self.top)
        plt.xlim(left=self.left)
        plt.xlim(right=self.right)
        plt.ylim(bottom=self.bottom)
        plt.ylim(top=self.top)

    def redraw(self):
        plt.draw()
        plt.pause(0.01) #is necessary for the plot to update for some reason

    @property
    def line1(self):
        return self.hl1.get_xdata(), self.hl1.get_ydata()

    @line1.setter
    def line1(self, xy):
        x, y = xy
        self.hl1.set_xdata(x)
        self.hl1.set_ydata(y)
        self.update_xylim(x, y)

    def extend_line1(self, x, y):
        old_x, old_y = self.line1
        self.line1 = (np.append(old_x, x), np.append(old_y, y))

    @property
    def line2(self):
        return self.hl2.get_xdata(), self.hl2.get_ydata()

    @line2.setter
    def line2(self, xy):
        x, y = xy
        self.hl2.set_xdata(x)
        self.hl2.set_ydata(y)
        self.update_xylim(x, y)

    def extend_line2(self, x, y):
        old_x, old_y = self.line2
        self.line2 = (np.append(old_x, x), np.append(old_y, y))


if __name__ == "__main__":

    gena = GraphAnimation('line1', 'line2')

    for i in range(200):
        gena.extend_line2([i], [math.sin(i / 10)])
        # gena.extend_line1([i], [math.cos(i / 10)])
        gena.redraw()

