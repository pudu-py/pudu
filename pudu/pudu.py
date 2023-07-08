from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
import copy

import masks as msk
import perturbation as ptn
import standards, error_handler 


class pudu:
    def __init__(self, x, y, pf, model=None):
        """
        `pudu` constructor.

        :type x: list
        :param x: Features (input) to be analyzed. Must have same format as 
            train and test descriptors with dientionality as 
            (batch, rows, columns, depth).
            
        :type y: int, float
        :param y: Targets (output) for `x`. It is a sclaar and not categorical
            for easier inclusion of regression problems. 
        
        :type pf: function
        :param pf: probability, or prediction, function of the algorithm. The input 
            must be `x` and the ouput a list of probabilities for each class (in
            case of classification algorithm). If the default function does not
            work this way (i.e.: needs a batch as input), it must be wrapped to
            do so. Please refer to the documentation's examples to see specific
            cases.
        
        :type model: Keras model
        :param model: Optional Keras model, only for `layer_activations` and 
            `unit_activation`.      
        """
        # Store the main parameters
        self.x = x
        self.y = y
        self.pf = pf

        # Optional parameters
        self.model = model

        # Main results
        self.imp = None
        self.spe = None
        self.syn = None

        # Activation results
        self.lac = None # layer activation
        self.uac = None # unit activations
        self.fac = None # feature activations
        self.icc = None

        # Normalized results are calculated automatically so if the user needs them
        self.imp_rel = None
        self.spe_rel = None
        self.syn_rel = None
        self.lac_rel = None
        self.uac_rel = None
        
        # Some error handling        
        error_handler.for_constructor(model, x, y)
        

    def importance(self, window=1, scope=None, evolution=None, padding='center', bias=0,
                    absolute=False, perturbation=ptn.Bidirectional(), mask=msk.All()):
        """
        Calculates the importance vector for the input feature.

        :type window: int
        :param window: feature width to be changeg each time.
            
        :type scope: tupple(int, int)
        :param scope: Starting and ending point of the analysis for each feature.
            If `None`, the all the vector is analysed. 

        :type evolution: int
        :param evolution: feature width to be changeg each time.
        
        :type padding: string 
        :param padding: Type of padding. If the legnth of `x` is not divisible 
            by `window` then padding is applyed. If `center`, then equal padding 
            to each side is applyed. If `right`, then paading to the right is 
            added and `window`starts from `0`. If `left`, padding to the left
            is applyied and `window` ends at length `x`. If perfet `center` is
            not possible, then ipadding left is added `1`.
        
        :type absolute: bool
        :param absolute: Weather or not the result is in absolute value or not. Default is `False`.
        """
        error_handler.for_params(window, scope, padding, absolute, 0, None, None)

        # Initial values
        sh = np.array(self.x).shape
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))

        scope, window, padd, evolution, total = standards.params_std(self.y, sh, scope, window, padding, evolution)

        p0 = self.pf(self.x)
        section = 1
        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:
                x_copy = copy.deepcopy(self.x)

                mask_val = mask.apply(section, total)

                if mask_val == 1:

                    row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                    row_idx, col_idx = row_idx + row, col_idx + col

                    temp, temp2 = perturbation.apply(x_copy, row, col, window, bias)

                    # print(np.shape(temp[0,:,:,0]), np.shape(self.x[0,:,:,0]))
                    # plt.plot(temp[0,0,:,0])
                    # plt.plot(self.x[0,0,:,0])
                    # plt.show()
                    # exit()

                    if temp2 is None:
                        val = self.pf(temp) - p0
                    else:
                        val = (self.pf(temp2) + self.pf(temp) - 2*p0) / 2

                    if absolute:
                        val = abs(val)

                    if np.shape(val):
                        d_temp[0, row:row+window[0], col:col+window[1], 0] = val[evolution]
                    else:
                        d_temp[0, row:row+window[0], col:col+window[1], 0] = val
                
                else:
                    pass

                section += 1

                col += window[1]
            row += window[0]

        self.imp = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.imp_rel = (d_temp - min_val) / (max_val - min_val)


    def speed(self, window=1, scope=None, evolution=None, padding='center',
                bias=0, absolute=False, mask=msk.All(), 
                perturbation=[ptn.Bidirectional(delta=.1), ptn.Bidirectional(delta=.2), ptn.Bidirectional(delta=.3)]):
        """
        Calculates the gradient of the importance. In other words, the slope
            of the curve formed by the importance at different values. This 
            indicates how fast a feature can change the result.
            
        :type window: int
        :param window: feature width to be changeg each time.
            
        :type scope: tupple(int, int)
        :param scope: Starting and ending point of the analysis for each feature.
            If `None`, the all the vector is analysed. 
        
        :type evolution: int
        :param evolution: feature width to be changeg each time.
        
        :type padding: string 
        :param padding: Type of padding. If the legnth of `x` is not divisible 
            by `window` then padding is applyed. If `center`, then equal padding 
            to each side is applyed. If `right`, then paading to the right is 
            added and `window`starts from `0`. If `left`, padding to the left
            is applyied and `window` ends at length `x`. If perfet `center` is
            not possible, then ipadding left is added `1`.

        :type absolute: bool
        :param absolute: Weather or not the result is in absolute value or not. Default is `False`.
        """
        error_handler.for_params(window, scope, padding, absolute, 0, None, None)

        # Initial values
        sh = np.array(self.x).shape
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        
        scope, window, padd, evolution, total = standards.params_std(self.y, sh, scope, window, padding, evolution)

        p0 = self.pf(self.x)
        section = 1
        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:
                x_copy = copy.deepcopy(self.x)

                mask_val = mask.apply(section, total)

                if mask_val == 1:

                    p = []

                    row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                    row_idx, col_idx = row_idx + row, col_idx + col
                    
                    for j in perturbation:
                        temp = self.x.copy()
                       
                        temp, temp2 = j.apply(x_copy, row, col, window, bias)

                        if temp2 is False:
                            val = self.pf(temp) - p0
                        else:
                            val = (self.pf(temp2) + self.pf(temp)) / 2

                        if absolute:
                            val = abs(val)
                        
                        if np.shape(val):
                            val = val[evolution]

                        p.append(val)

                    var_x = [i for i in range(len(p))]
                    var_y = [i for i in p]
                            
                    d_temp[0, row:row+window[0], col:col+window[1], 0] = np.polyfit(var_x, var_y, 1)[0]

                else:
                    pass    
                
                section += 1

                col += window[1]
            row += window[0]

        self.spe = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.spe_rel = (d_temp - min_val) / (max_val - min_val)

    
    def synergy(self, window=1, inspect=0, scope=None, absolute=False, bias=0,
                    evolution=None, padding='center', perturbation=ptn.Bidirectional(), mask=msk.All()):
        """
        Calculates the synergy between features.
        
        :type delta: float
        :param delta: maximum variation to apply to each feature.
            
        :type window: int
        :param window: feature width to be changeg each time.
            
        :type scope: tupple(int, int)
        :param scope: Starting and ending point of the analysis for each feature.
            If `None`, the all the vector is analysed. 
        
        :type evolution: int
        :param evolution: feature width to be changeg each time.
        
        :type padding: string 
        :param padding: Type of padding. If the legnth of `x` is not divisible 
            by `window` then padding is applyed. If `center`, then equal padding 
            to each side is applyed. If `right`, then paading to the right is 
            added and `window`starts from `0`. If `left`, padding to the left
            is applyied and `window` ends at length `x`. If perfet `center` is
            not possible, then ipadding left is added `1`.
        
        :type absolute: bool
        :param absolute: Weather or not the result is in absolute value or not. Default is `False`.
        """
        error_handler.for_params(window, scope, padding, absolute, inspect, None, None)

        # Initial values
        sh = np.array(self.x).shape
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        
        scope, window, padd, evolution, total = standards.params_std(self.y, sh, scope, window, padding, evolution)

        # Position to range of the desired area to calculate synergy from
        if len(np.array(inspect).shape) == 0:
            if sh[1] == 1:
                inspect = (0, int(window[1]*inspect + padd[1][0] + scope[1][0]))
            else:
                inspect = (window[0]*inspect + padd[0][0] + scope[0][0], 
                           window[1]*inspect + padd[1][0] + scope[1][0])
                
        x_copy = copy.deepcopy(self.x)
        base, base2 = perturbation.apply(x_copy, inspect[0], inspect[1], window, bias)

        pb0 = self.pf(base)
        pb2 = self.pf(base2)

        section = 1
        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:
                x_copy = copy.deepcopy(self.x)

                mask_val = mask.apply(section, total)

                if mask_val == 1:

                    if inspect[0] == row and inspect[1] == col:
                        pass
                    else:
                        row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                        row_idx, col_idx = row_idx + row, col_idx + col

                        temp, temp2 = perturbation.apply(x_copy, row, col, window, bias)

                        if temp2 is False:
                            val = self.pf(temp) - pb0
                        else:
                            val = (self.pf(temp2) + self.pf(temp) - pb0 - pb2) / 2

                        if absolute:
                            val = abs(val)

                        if np.shape(val):
                            d_temp[0, row:row+window[0], col:col+window[1], 0] = val[evolution]
                        else:
                            d_temp[0, row:row+window[0], col:col+window[1], 0] = val
                else:
                    pass

                section += 1

                col += window[1]
            row += window[0]

        self.syn = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.syn_rel = (d_temp - min_val) / (max_val - min_val)


    def activations(self, layer=0, slope=0, p=0.005, window=1, scope=None, bias=0,
                        padding='center', perturbation=ptn.Bidirectional(), mask=msk.All()):
        """
        Counts the unit activations in the selected `layer` of a `Keras` model according 
            to change in the feature.
        
        :type layer: int
        :param layer: Position number within the keras model to be analyzed. Use `model.summary()`
            to see exactly the position of the desired layer.

        :type slope: float
        :param slope: Default is 0 (`relu`).

        :type window: int
        :param window: feature width to be changeg each time.
            
        :type scope: tupple(int, int)
        :param scope: Starting and ending point of the analysis for each feature.
            If `None`, the all the vector is analysed. 
        
        :type padding: string 
        :param padding: Type of padding. If the legnth of `x` is not divisible 
            by `window` then padding is applyed. If `center`, then equal padding 
            to each side is applyed. If `right`, then paading to the right is 
            added and `window`starts from `0`. If `left`, padding to the left
            is applyied and `window` ends at length `x`. If perfet `center` is
            not possible, then ipadding left is added `1`.
        """
        error_handler.for_params(window, scope, padding, False, 0, layer, p)
        
        # Initial values
        sh = np.array(self.x).shape
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        o_b, o_h, o_w, o_d = sh[0], sh[1], sh[2], sh[3]
        
        scope, window, padd, evolution, total = standards.params_std(self.y, sh, scope, window, padding, None)

        # Keras
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)

        x = np.squeeze(self.x, axis=0) if self.x.shape[1] == 1 else self.x
        activations = activation_model.predict(x, verbose=0)
        p0 = activations[layer] # raw activation values for the image

        # index_array = [[] for _ in range(len(p0.flatten()))] # to store unit idx and coordinates

        act_count = []
        act_idx_count = 1
        section = 1
        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:
                x_copy = copy.deepcopy(self.x)
                
                mask_val = mask.apply(section, total)

                if mask_val == 1:

                    row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                    row_idx, col_idx = row_idx + row, col_idx + col

                    temp, temp2 = perturbation.apply(x_copy, row, col, window, bias)

                    if temp2 is None:
                        temp = np.squeeze(temp, axis=0) if temp.shape[1] == 1 else temp

                        activations = activation_model.predict(temp, verbose=0)
                        activations = activations[layer]
                        activations = activations-p0

                    else:
                        temp = np.squeeze(temp, axis=0) if temp.shape[1] == 1 else temp
                        temp2 = np.squeeze(temp2, axis=0) if temp2.shape[1] == 1 else temp2

                        activations = activation_model.predict(temp, verbose=0)
                        p1 = activations[layer]

                        activations = activation_model.predict(temp2, verbose=0)
                        p2 = activations[layer]

                        activations = (p1 + p2 - 2*p0) / 2

                    act_count.append(activations.flatten())

                    d_temp[0, row:row+window[0], col:col+window[1], 0] = act_idx_count
                    act_idx_count += 1

                else:
                    pass

                section += 1

                col += window[1]
            row += window[0]

        """
        At this point 'act_count' has dims. (n, k), where n is the number
        of times the kernel fits in the input and 'k' is the number of of units.
        So now we calculate the quantiles and store the values above the quantile
        and their indices (to what feature change they belong, basically).
        Then we apply leakyrelu (relu as default)
        Then we count the number of activations
        """

        idx_w_vals = []
        act_count = np.array(act_count)

        act_count = np.transpose(act_count)
        act_count = np.where(act_count > 0, act_count, slope * act_count) # new
        act_count = np.array(act_count)

        quantiles = [np.quantile(i, p) for i in act_count]
        quantiles = np.array(quantiles)

        mask = (act_count != 0) & (act_count >= quantiles[:, np.newaxis])
        i_vals, j_vals = np.nonzero(mask)
        a_vals = act_count[i_vals, j_vals]
        idx_w_vals = np.column_stack([i_vals, j_vals, a_vals]).tolist()
   
        # 'acts' is the activation value, but if it is in this 
        # varibales then it is considered activated anyways
        # here we have all the info we need:
        units, feats, acts = np.transpose(idx_w_vals)
        feats += 1

        feats = np.array(feats).astype(int)
        counts_feats = np.bincount(feats)
        if len(counts_feats) < act_idx_count:
            counts_feats = np.pad(counts_feats, (0, act_idx_count - len(counts_feats)), 'constant')

        units = np.array(units).astype(int)
        counts_units = np.bincount(units)
        if len(counts_units) < len(quantiles):
            counts_units = np.pad(counts_units, (0, len(quantiles) - len(counts_units)), 'constant')

        # contar partes del mapping que no son zero (no padding)
        feats = np.array(feats)
        unique, counts = np.unique(feats, return_counts=True)
        feats_counts = dict(zip(unique, counts))
        feats_counts = defaultdict(int, feats_counts)
        vfunc = np.vectorize(feats_counts.__getitem__)
        mask = d_temp[0, :, :, 0] != 0
        d_temp[0, :, :, 0][mask] = vfunc(d_temp[0, :, :, 0][mask])

        self.fac = counts_feats

        self.lac = d_temp # this shows the activation mapping, or activations per feature
        max_val, min_val = d_temp.max(), d_temp.min()
        self.lac_rel = (d_temp - min_val) / (1 if (max_val - min_val) == 0 else (max_val - min_val))
        
        self.uac = counts_units # new
        max_val, min_val = counts_units.max(), counts_units.min()
        self.uac_rel = (counts_units - min_val) / (1 if (max_val - min_val) == 0 else (max_val - min_val))

        un_fe = defaultdict(list)
        for u, f in zip(units, feats):
            un_fe[u].append(f)
        un_fe = [un_fe[i] for i in range(max(units) + 1)]

        result = []
        for i, sublist in enumerate(un_fe):
            if sublist: # if not empty
                counts = Counter(sublist) # reps. of each number
                most_common_num, num_repetitions = counts.most_common(1)[0] # most reps.
                result.append([i, most_common_num, num_repetitions])
        
        return feats, units


    def relatable(self, layer=0, slope=0, p=0.005, window=1, scope=None, bias=0,
                    padding='center', perturbation=ptn.Bidirectional(), mask=msk.All()):
        """
        This function generates an activation report for each set of coordinates in `x` and `y`. 

        :type layer: int
        :param layer: Specifies the layer of the model for which the activation report is to be generated. 
                    Default is 0.

        :type slope: int or float
        :param slope: 

        :type p: float
        :param p: Specifies the p-value threshold for significance testing of activations. Default is 0.005.

        :type window: int
        :param window: Specifies the size of the window for the activation function. Default is 1.

        :type scope: str
        :param scope: Specifies the scope of the activations. Possible values are 'global' and 'local'. 
                    Default is None.

        :type padding: str
        :param padding: Specifies the padding strategy for the activations. Default is 'center'.

        :type kwargs: dict
        :param kwargs: Additional keyword arguments passed to the activation function.
        """
        # hay que cambiar self.x e .y para iterar. Se guarda y luego
        # se cambia por cada integrante. Al final se vuelve a la forma
        # original si se desea usar denuevo
        s_x, s_y = self.x, self.y

        master = []
        # master = np.array()
        for x, y in zip(s_x, s_y):
            x = np.expand_dims(x, 0)
            self.x, self.y = x, y

            feats, units = self.activations(layer, slope, p, window, scope, bias, padding,
                                            perturbation=perturbation, mask=mask)

            master.extend((i, j) for i, j in zip(feats, units))

            feats = np.array(feats)
            units = np.array(units)
            master = np.column_stack((feats, units))
                        
            master = master.tolist()

        master = [tuple(arr) for arr in master]
        counts = Counter(master)
        result = [[j, count, i] for (i, j), count in counts.items()]
        result = np.transpose(result)

        self.x, self.y = s_x, s_y
        self.icc = result


    def preview(self, window=1, scope=None, padding='center', axis=None, show_data=True,
                    title='Preview', xlabel='Feature', ylabel='Intensity', xticks=None, 
                    yticks=[], cmap='Greens', font_size=15, figsize=(14, 4), bold=False,
                    mask = msk.All()):
        """
        Plots an approximate preview of the sections, areas, or mask to be analyzed over the data
            before executing. It is particularly useful to check if the parameters are
            correct, especially if the user expects long runtimes.
        
        :type feature: list
        :param feature: feature analyzed or any that the user whant to plot against.
            Normally you want it to be `self.x`.
        
        :type window: int
        :param window: feature width to be analyzed.

        :type scope: tuple
        :param scope: feature width to be analyzed.

        :type padding: str
        :param padding: Type of padding. If the legnth of `x` is not divisible 
            by `window` then padding is applyed. If `center`, then equal padding 
            to each side is applyed. If `right`, then paading to the right is 
            added and `window`starts from `0`. If `left`, padding to the left
            is applyied and `window` ends at length `x`. If perfet `center` is
            not possible, then ipadding left is added `1`.
        
        :type mask: list
        :param mask: Mask to be applied to the data. If `None`, no mask will be
            applied.

        :type axis: list
        :param axis: X-axis for the plot. If `None`, it will show the pixel count.

        :type show_data: bool
        :param show_data: If `True`, it will plot the data. If `False`, it will
            plot the mask.

        :type title: str
        :param title: Title of the plot.

        :type xlabel: str
        :param xlabel: X-axis label.

        :type ylabel: str
        :param ylabel: Y-axis label.

        :type xticks: list
        :param xticks: X-axis ticks.

        :type yticks: list
        :param yticks: Y-axis ticks.

        :type cmap: str
        :param cmap: Colormap to be used.

        :type font_size: int
        :param font_size: Font size for the plot.

        :type figsize: tuple
        :param figsize: Figure size for the plot.

        :type bold: Boolean
        :param bold: To make the limit lines bolder. Default is 'False'.
        """
        error_handler.for_params(window, scope, padding, False, 0, None, None)

        # Initial values
        image = [] # this will be the preview image
        sh = np.array(self.x).shape
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        countour = np.zeros((sh[0], sh[1], sh[2], sh[3]))
                
        scope, window, padd, evolution, total = standards.params_std(self.y, sh, scope, window, padding, None)

        section = 1
        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:
                mask_val = mask.apply(section, total)

                if mask_val == 1:
                    countour[0, row, col, 0] = 1
                    d_temp[0, row:row+window[0], col:col+window[1], 0] = 1

                    if sh[1] > 1:
                        for i in range(sh[1]):
                            countour[0, i, col, 0] = 1
                        for i in range(sh[2]):
                            countour[0, row, i, 0] = 1
                else:
                    pass
                
                section += 1
                col += window[1]
            row += window[0]

        dims = np.array(d_temp).shape
        image = d_temp[0,:,:,0]
        countour = countour[0,:,:,0]

        feature = self.x
        feature = np.array(feature)[0,:,:,0]

        if dims[1] > 1:
            rows, cols = dims[1], dims[2]
            ext = [0, cols, 0, rows]
        else:
            rows, cols = 1, len(feature)
        
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

        # just to make bolder lines
        if bold == True:
            for i in range(len(countour)):
                for j in range(len(countour[i])):
                    try:
                        if countour[i][j] == 1 and countour[i][j+1] == 0 and countour[i][j-1] == 0:
                                countour[i][j-1] = countour[i][j+1] = 1

                        if dims[1] > 1 and countour[i][j] == 1 and countour[i+1][j] == 0 and countour[i-1][j] == 0:
                                countour[i-1][j] = countour[i+1][j] = 1
                    except IndexError:
                        print('Index out of bounds. Proceed normally.')

        plt.imshow(countour, cmap='binary', aspect="auto", 
                   interpolation='nearest', extent=ext, alpha=1)
        
        plt.imshow(image, cmap=cmap, aspect="auto", 
                   interpolation='nearest', extent=ext, alpha=0.5)
        
        plt.title(title) 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yticks(yticks)

        if xticks:
            plt.xticks(axis, xticks, rotation='vertical')
        
        plt.show()  
