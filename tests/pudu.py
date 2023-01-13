import matplotlib.pyplot as plt
import spectrapepper as spep
import numpy as np
import copy


class pudu:
    def __init__(self, x, y, pf):
        """
        `pudu` constructor.

        :type x: list
        :param x: Features (input) to be analyzed. Must have same format as 
            train and test descriptors.
            
        :type y: int, float
        :param y: Targets (output) of `x`. 
        
        :type pf: function
        :param pf: probability, or predict, function of the algorithm. The input 
            must be `x` and the ouput a list of probabilities for each class (in
            case of classification algorithm). If the default function does not
            work this way (i.e.: needs a batch as input), it must be wrapped to
            do so. Please refer to the documentation's example about this.
        """
        # Store the main parameters
        self.x = x
        self.y = y
        self.pf = pf
        
        # Main results
        self.imp = None
        self.spe = None
        self.syn = None

        # Normalized results are calculated automatically so if the user needs them
        self.imp_norm = None
        self.spe_norm = None
        self.syn_norm = None
        
        # Some dimension error handling
        if len(np.array(x).shape) != 4:
            raise ValueError(f"Expected array to have rank 4 (batch, rows, columns, depth). Got array with shape: %s" % str(np.array(x).shape))
        
        if len(np.array(y).shape) != 0:
            raise ValueError(f"Expected integer. Got array with shape: %s" % str(np.array(y).shape))
        

    def importance(self, delta=0.1, window=1, scope=None, calc='absolute', 
                       evolution=None, padding='center', bias=0):
        """
        Calculates the importance vector for the input feature.

        :type delta: float
        :param delta: maximum variation to apply to each features.
            
        :type window: int
        :param window: feature width to be changeg each time.
            
        :type scope: tupple(int, int)
        :param scope: Starting and ending point of the analysis for each feature.
            If `None`, the all the vector is analysed. 
        
        :type calc: string
        :param calc: Can be `absolute` or `relative`. If `absolute` the importance
            is calculated using the average of the absolute vbalues. If `relative`
            the it uses the real value for the average.
        
        :type evolution: int
        :param evolution: feature width to be changeg each time.
        
        :type padding: string 
        :param padding: Type of padding. If the legnth of `x` is not divisible 
            by `window` then padding is applyed. If `center`, then equal padding 
            to each side is applyed. If `right`, then paading to the right is 
            added and `window`starts from `0`. If `left`, padding to the left
            is applyied and `window` ends at length `x`. If perfet `center` is
            not possible, then ipadding left is added `1`.
        """
        # Initial values
        sh = np.array(self.x).shape
        x_copy = copy.deepcopy(self.x)
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        padd = [[0, 0], [0, 0]]
        
        # Dim. std.
        if scope is None:
            scope = (0, sh[2])
        
        if len(np.array(window).shape) == 0:
            if sh[1] == 1:
                window = (1, window)
            else:
                window = (window, window)
        
        if len(np.array(scope).shape) == 1:
            if sh[1] == 1:
                scope = ((0, 1), scope)
            else:
                scope = (scope, scope)
        
        if len(np.array(padding).shape) == 0:
            padding = (padding, padding)
        
        if evolution is None:
            evolution = int(self.y)
        
        # Padding
        for i in range(2):
            comp = int((scope[i][1]-scope[i][0])%window[i])
            if comp > 0:
                padd[i] = self.calc_pad(padding[i], comp)

        p0 = self.pf(self.x)

        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:
                p = [p0, -1, 1]
                for j in range(1, 3):
                    row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                    row_idx, col_idx = row_idx + row, col_idx + col

                    temp = x_copy.copy()
                    temp[0, row_idx, col_idx, 0] = temp[0, row_idx, col_idx, 0] * (1 - delta * p[j]) + bias
                    p[j] = self.pf(temp)
                
                if calc == 'absolute':
                    val = (abs(p[0]-p[2]) + abs(p[0]-p[1]))/2
                elif calc == 'relative':
                    val = ((p[2]-p[0]) + (p[1]-p[0]))/2
                
                if np.shape(val):
                    d_temp[0, row:row+window[0], col:col+window[1], 0] = val[evolution]
                else:
                    d_temp[0, row:row+window[0], col:col+window[1], 0] = val

                col += window[1]
            row += window[0]

        self.imp = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.imp_norm = (d_temp - min_val) / (max_val - min_val)


    def speed(self, delta=0.1, window=1, scope=None, calc='absolute', 
                       evolution=None, padding='center', steps=3, bias=0):
        """
        Calculates the gradient of the iomportance. In other owrds, the slope
            of the importance at different values. This indicates how fast a
            feature can change the result.

        :type delta: float
        :param delta: maximum variation to apply to each features.
            
        :type window: int
        :param window: feature width to be changeg each time.
            
        :type scope: tupple(int, int)
        :param scope: Starting and ending point of the analysis for each feature.
            If `None`, the all the vector is analysed. 
        
        :type calc: string
        :param calc: Can be `absolute` or `relative`. If `absolute` the importance
            is calculated using the average of the absolute vbalues. If `relative`
            the it uses the real value for the average.
        
        :type evolution: int
        :param evolution: feature width to be changeg each time.
        
        :type padding: string 
        :param padding: Type of padding. If the legnth of `x` is not divisible 
            by `window` then padding is applyed. If `center`, then equal padding 
            to each side is applyed. If `right`, then paading to the right is 
            added and `window`starts from `0`. If `left`, padding to the left
            is applyied and `window` ends at length `x`. If perfet `center` is
            not possible, then ipadding left is added `1`.
        """
        # Initial values
        sh = np.array(self.x).shape
        x_copy = copy.deepcopy(self.x)
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        padd = [[0, 0], [0, 0]]
        
        # Dim. std.
        if scope is None:
            scope = (0, sh[2])
        
        if len(np.array(window).shape) == 0:
            if sh[1] == 1:
                window = (1, window)
            else:
                window = (window, window)
        
        if len(np.array(scope).shape) == 1:
            if sh[1] == 1:
                scope = ((0, 1), scope)
            else:
                scope = (scope, scope)
        
        if len(np.array(padding).shape) == 0:
            padding = (padding, padding)
        
        if evolution is None:
            evolution = int(self.y)
            
        p0 = self.pf(self.x) 
        
        # Padding
        for i in range(2):
            comp = int((scope[i][1]-scope[i][0])%window[i])
            if comp > 0:
                padd[i] = self.calc_pad(padding[i], comp)
        
        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:
                p = [p0] + [i for i in range(1, steps + 1)]

                for j in range(1, steps + 1):
                    temp = x_copy.copy()
                    temp[0, row:row+window[0], col:col+window[1], 0] = temp[0, row:row+window[0], col:col+window[1], 0] * (1 - delta * p[j]) + bias
                    p[j] = self.pf(temp)
                
                var_x = [i for i in range(steps + 1)]
                var_y = [i[evolution] for i in p]
                
                slope = np.polyfit(var_x, var_y, 1)[0]

                d_temp[0, row:row+window[0], col:col+window[1], 0] = slope
            
                col += window[1]
            row += window[0]

        self.spe = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.spe_norm = (d_temp - min_val) / (max_val - min_val)

    
    def synergy(self, delta=0.1, window=1, inspect=0, scope=None, calc='absolute', 
                    evolution=None, padding='center', bias=0, mask=None):
        """
        Calculates the synergy between features.
        
        :type delta: float
        :param delta: maximum variation to apply to each feature.
            
        :type window: int
        :param window: feature width to be changeg each time.
            
        :type scope: tupple(int, int)
        :param scope: Starting and ending point of the analysis for each feature.
            If `None`, the all the vector is analysed. 
        
        :type calc: string
        :param calc: Can be `absolute` or `relative`. If `absolute` the importance
            is calculated using the average of the absolute vbalues. If `relative`
            the it uses the real value for the average.
        
        :type evolution: int
        :param evolution: feature width to be changeg each time.
        
        :type padding: string 
        :param padding: Type of padding. If the legnth of `x` is not divisible 
            by `window` then padding is applyed. If `center`, then equal padding 
            to each side is applyed. If `right`, then paading to the right is 
            added and `window`starts from `0`. If `left`, padding to the left
            is applyied and `window` ends at length `x`. If perfet `center` is
            not possible, then ipadding left is added `1`.
        """
        # Initial values
        sh = np.array(self.x).shape
        x_copy = copy.deepcopy(self.x)
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        
        padd = [[0, 0], [0, 0]]
        
        # Dim. std.
        if scope is None:
            scope = (0, sh[2])
        
        if len(np.array(window).shape) == 0:
            if sh[1] == 1:
                window = (1, window)
            else:
                window = (window, window)

        if len(np.array(scope).shape) == 1:
            if sh[1] == 1:
                scope = ((0, 1), scope)
            else:
                scope = (scope, scope)
        
        if len(np.array(padding).shape) == 0:
            padding = (padding, padding)
        
        if evolution is None:
            evolution = int(self.y)
         
        # Padding
        for i in range(2):
            comp = int((scope[i][1]-scope[i][0])%window[i])
            if comp > 0:
                padd[i] = self.calc_pad(padding[i], comp)
        
        # Position to range of the desired area to calculate synergy from
        if len(np.array(inspect).shape) == 0:
            if sh[1] == 1:
                inspect = (0, int(window[1]*inspect + padd[1][0] + scope[1][0]))
            else:
                inspect = (window[0]*inspect + padd[0][0] + scope[0][0], 
                           window[1]*inspect + padd[1][0] + scope[1][0])
        
        base = copy.deepcopy(x_copy)
        for i in range(inspect[0], inspect[0]+window[0]):
            for j in range(inspect[1], inspect[1]+window[1]):
                base[0][i][j][0] = x_copy[0][i][j][0]*(1+delta)
        
        b = self.pf(base) # this value is baseline, set to 0
        p0 = self.pf(self.x)

        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:
                if inspect[0] == row and inspect[1] == col:
                    pass  # Skip the current iteration
                else:
                    p = [p0, -1, 1]  # -1 and 1 for sign mult., replaced after
                    for j in range(1, 3):
                        temp = x_copy.copy()
                        temp[0, row:row+window[0], col:col+window[1], 0] = temp[0, row:row+window[0], col:col+window[1], 0] * (1 - delta * p[j]) + bias
                        p[j] = self.pf(temp) - p0  # p0 is baseline
                        
                    if calc == 'absolute':
                        val = (abs(p[0] - p[2]) + abs(p[0] - p[1])) / 2
                    elif calc == 'relative':
                        val = ((p[2] - p[0]) + (p[1] - p[0])) / 2
                    
                    d_temp[0, row:row+window[0], col:col+window[1], 0] = val[evolution]

                col += window[1]
            row += window[0]

        self.syn = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.syn_norm = (d_temp - min_val) / (max_val - min_val)
      
    
    def plot(self, feature, image, axis=None, show_data=True, title='Importance', 
        xlabel='Feature', ylabel='Intensity', xticks=None, yticks=[], cmap='plasma',
        font_size=15, figsize=(14, 4)):
        """
        Easy plot function for `importance`, `speed`, or `synergy`. It shows the analyzed
            feature `feature` with a colormap overlay indicating the result along with
            a colorbar. Works for both vectors and images.
    
        :type feature: list
        :param feature: feature analyzed or any that the user whant to plot against.
            Normally you want it to be `self.x`.

        :type image: list
        :param image: Result you want to plot. Either `self.imp`, `self.syn`, etc...

        :type axis: list
        :param axis: X-axis for the plot. If `None`, it will show the pixel count.
        
        :type show_data: bool
        :param show_data: . Default is `True`.

        :type title: str
        :param title: Title for the plot. Default is `Importnace`.
            
        :type xlabel: str
        :param xlabel: X-axis title. Default is `Feature`.
            
        :type ylabel: str
        :param ylabel: Y-axis title. Default is `Intensity`
        
        :type xticks: list
        :param xticks: Ticks to display on the graph. Default is `None`.

        :type yticks: list
        :param yticks: Ticks to display on the graph. Default is `[]`.

        :type cmap: string
        :param cmap: colormap for `image` according to the availability in `matplotlib`.
            Default is `plasma`.

        :type font_size: int
        :param font_size: Font size for all text in the plot. Default is `15`.
            
        :type figsize: tuple
        :param figsize: Size of the figure. Default is `(14, 4)`.
        """

        dims = np.array(image).shape

        image = np.array(image)[0,:,:,0]
        feature = np.array(feature)[0,:,:,0]
        
        if dims[1] > 1:
            rows, cols = dims[1], dims[2]
            ext = [0, cols, 0, rows]
        else:
            rows, cols = 1, len(image)
                    
            if axis is None:
                axis = [i for i in range(len(feature[0]))]
                ext = [0, len(feature[0]), min(feature[0]), max(feature[0])]
            else:
                ext = [min(axis), max(axis), min(feature[0]), max(feature[0])]
        
        plt.rc('font', size=font_size)
        plt.figure(figsize=figsize)
        if dims[1] > 1 and show_data:
            plt.imshow(feature, cmap='binary', aspect="auto", 
                       interpolation='nearest', extent=ext, alpha=1)
        elif dims[1] == 1 and show_data:
            plt.plot(axis, feature[0], 'k')
        plt.imshow(image, cmap=cmap, aspect="auto", 
                   interpolation='nearest', extent=ext, alpha=0.5)
        plt.title(title) 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yticks(yticks)

        if xticks:
            plt.xticks(axis, xticks, rotation='vertical')

        plt.colorbar()
        plt.show()  


    def save(self, name='pudu_data.txt', transpose=False):
        """
        Saves all the vectors in a `.txt` file to make it easier to export the data to 
            other software, scripts, or wererver the user needs. In other words, this only saves
            the results as there is no model to save. This is particularly useful if other programs, 
            such as `Origin Pro`, are used to generate figures or further analysis.
        
        :type name: string 
        :param name: Name of the file to be saved. Default is `pudu_data.txt`.

        :type transpose: boolean 
        :param transpose: If `True`, data will be tranpose to columns. Deafult is `False`, each
            vector will be saved in each row.
        """
        sh = np.array(self.x).shape
        data = []
        
        for variable in ('imp', 'imp_norm', 'spe', 'spe_norm', 'syn', 'syn_norm'):
            if getattr(self, variable) is not None:
                data.append(getattr(self, variable)[0, :, :, 0])
        
        print(np.shape(data))

        if sh[1] == 1: 
            data = [item[0] for item in data]
        else:
            data2 = []
            for i in range(len(data)):
                for j in range(sh[1]):
                    data2.append(data[i][j])

            data = data2

        if transpose:
            data = np.transpose(data)
        
        np.savetxt(name, data, fmt='%s')


    def calc_pad(self, t, comp):
        """
        Calculate padding for the given type.

        :type t: str
        :param t: Type of padding. Can be 'center', 'left', or 'right'.

        :rtype: list
        :returns: A list of two integers representing the padding for 
            the left and right sides.
        """
        if t == 'center':
            if comp % 2 == 0:  # even number
                pad = [int(comp / 2), int(comp / 2)]
            else:  # if odd number, left gets the +1
                pad = [int(np.ceil(comp / 2)), int(np.floor(comp / 2))]
        elif t == 'left':
            pad = [comp, 0]
        elif t == 'right':
            pad = [0, comp]
        else:
            raise ValueError(f"Invalid padding type '{t}'. Valid types are 'center', 'left', and 'right'.")
        
        return pad
