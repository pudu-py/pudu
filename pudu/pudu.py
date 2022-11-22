import matplotlib.pyplot as plt
import spectrapepper as spep
import pandas as pd
import numpy as np
import math
import copy


class pudu:
    def __init__(self, x, y, pf):
        """
        Description.

        :type : 
        :param :
          
        :returns:
        :rtype:
        """
        self.x = x
        self.y = y
        self.pf = pf
        self.imp = None
        
    def feed(self, d=0.1, window=1, padding='center'):
        """
        Description.

        :type : 
        :param :
          
        :returns:
        :rtype:
        """
        div_res = len(self.x)%window
        comp = -1
        pad_l, pad_r = 0, 0 # initial value is 0
        if div_res != 0:
            comp = window-div_res # complement for integer division
        
        # if comp >= 0:
        #     if padding == 'center':
        #         if comp%2 == 0: # even number
        #             pad_l, pad_r = comp/2, comp/2
        #         else: # if odd number, left always gets the +1
        #             pad_l, pad_r = math.ceil(comp/2), math.floor(comp/2)
        #     if padding == 'left':
        #         pad_l, pad_r = comp, 0
        #     if padding == 'right':
        #         pad_l, pad_r = 0, comp
        # else:
        #     if padding == 'center':
        #         pad_l, pad_r = (window-1)/2, (window-1)/2
        #     if padding == 'left':
        #         pad_l, pad_r = window, 0
        #     if padding == 'right':
        #         pad_l, pad_r = 0, comp
        
        d_temp = [0 for _ in range(len(self.x))]
        
        i = 0
        while i<=len(self.x)-window:
        # for i in range(0+pad_l, len(x)-pad_r-window+1):
            
            temp = copy.deepcopy(self.x)
            p0 = self.pf([temp])[int(self.y)]
            
            for j in range(window):
                temp[i+j] = self.x[i+j]*(1-d)
            p1 = self.pf([temp])[int(self.y)]
            
            temp = copy.deepcopy(self.x)
            
            for j in range(window):
                temp[i+j] = self.x[i+j]*(1+d)
            p2 = self.pf([temp])[int(self.y)]
            
            val = (abs(p0-p2) + abs(p0-p1))/2
            
            for j in range(window):
                d_temp[i+j] = val
            i += window
        
        self.imp = d_temp
        
    def plot(self, axis=None, title='Importance', xlabel='Feature', 
             ylabel='Intensity', font_size=15, figsize=(14, 4)):
        """
        Description.

        :type : 
        :param :
          
        :returns:
        :rtype:
        """
        ext = []
        if axis is None:
            axis = [i for i in range(len(self.x))]
            ext = [0, len(self.x), min(self.x), max(self.x)]
        else:
            ext = [min(axis), max(axis), min(self.x), max(self.x)]
            
        plt.rc('font', size=font_size)
        plt.figure(figsize=figsize)
        plt.imshow(np.expand_dims(self.imp, 0), cmap='Greens', aspect="auto", 
                   interpolation='nearest', extent=ext, alpha=1)
        plt.plot(axis, self.x, 'k')
        plt.title(title) 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yticks([])
        plt.colorbar()
        plt.show()
        
    def average(self):
        None
    def typical(self):
        None
    def representative(self):
        None
        
    def normalize(self):
        """
        Description.

        :type : 
        :param :
          
        :returns:
        :rtype:
        """
        self.imp = spep.normtomax(self.imp, to=1, zeromin=True)
        
    def save(self):
        None
