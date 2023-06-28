from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import spectrapepper as spep
import numpy as np
import copy
from . import perturbation, masks

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

        # Optional results
        self.lac = None # layer activation
        self.uac = None # unit activation
        self.aso = None # unit associacation with feature

        # Normalized results are calculated automatically so if the user needs them
        self.imp_norm = None
        self.spe_norm = None
        self.syn_norm = None
        self.lac_norm = None
        self.uac_norm = None
        
        # Some error handling
        if model is None:
            pass
        if not isinstance(model, (Sequential, Model)):
            raise ValueError("Expected `model` to be a keras model")

        if np.array(x).shape[0] > 1 and (not isinstance(y, list)):
            raise ValueError("Expected `x` and `y` to have the same length."
                                "Got `x` with length (batch size): %s" % str(np.array(x).shape[0]) %
                                "and `y` is the scalar: %s " % str(y))
        
        if isinstance(y, list) and np.array(x).shape[0] != len(y):
            raise ValueError("Expected `x` and `y` to have the same length."
                                "Got `x` with length (batch size): %s" % str(np.array(x).shape[0]) %
                                "and `y` with length: %s " % str(len(y)))

        if len(np.array(x).shape) != 4:
            raise ValueError("Expected array to have rank 4 (batch, rows, columns, depth)."
                                "Got array with shape: %s" % str(np.array(x).shape))
        
        if len(np.array(y).shape) != 0:
            raise ValueError("Expected integer. Got array with shape: %s" % str(np.array(y).shape))
        

    def importance(self, window=1, scope=None, evolution=None, padding='center', 
                    absolute=False, **kwargs):
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
        error_handling(window, scope, padding, absolute, 0, [1, 2, 3], None, None)

        # Initial values
        sh = np.array(self.x).shape
        x_copy = copy.deepcopy(self.x)
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        
        scope, window, padding, evolution = params_std(self.y, sh, scope, window, padding, evolution)
        padd = calc_pad(padding, scope, window)
        mask_array = masks.function(sh=sh, padd=padd, scope=scope, window=window, **kwargs)

        p0 = self.pf(self.x)
        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:

                if mask_array[0][row][col][0] == 1:

                    row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                    row_idx, col_idx = row_idx + row, col_idx + col
                    
                    temp, temp2 = perturbation.function(x=x_copy, row=row_idx, col=col_idx, **kwargs)

                    if temp2 is False:
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

                col += window[1]
            row += window[0]

        self.imp = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.imp_norm = (d_temp - min_val) / (max_val - min_val)


    def speed(self, window=1, scope=None, evolution=None, padding='center', 
                steps=[0,0.1,0.2], absolute=False, **kwargs):
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

        :type steps: list
        :param steps: Contains the different values at which the importance will be measured.
            In other words, the values at which the feature will be changed (feature*steps)
            before applyin the perturbation function (perturbation(feature*steps)). This means
            it orks somewhat differently than `importnace` and `synergy`.
        """
        error_handling(window, scope, padding, absolute, 0, steps, None, None)

        # Initial values
        sh = np.array(self.x).shape
        x_copy = copy.deepcopy(self.x)
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        
        scope, window, padding, evolution = params_std(self.y, sh, scope, window, padding, evolution)
        padd = calc_pad(padding, scope, window)
        mask_array = masks.function(sh=sh, padd=padd, scope=scope, window=window, **kwargs)

        p0 = self.pf(self.x) 
        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:

                if mask_array[0][row][col][0] == 1:

                    p = []

                    row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                    row_idx, col_idx = row_idx + row, col_idx + col

                    for j in steps:
                        temp = self.x.copy()
                        temp[0, row, col, 0] = (x_copy[0, row, col, 0]) * j
                        temp, temp2 = perturbation.function(x=temp, row=row_idx, col=col_idx, **kwargs)
                        
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

                col += window[1]
            row += window[0]

        self.spe = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.spe_norm = (d_temp - min_val) / (max_val - min_val)

    
    def synergy(self, delta=0.1, window=1, inspect=0, scope=None, absolute=False,
                    evolution=None, padding='center', **kwargs):
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
        error_handling(window, scope, padding, absolute, inspect, [1, 2, 3], None, None)

        # Initial values
        sh = np.array(self.x).shape
        x_copy = copy.deepcopy(self.x)
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        
        scope, window, padding, evolution = params_std(self.y, sh, scope, window, padding, evolution)
        padd = calc_pad(padding, scope, window)
        mask_array = masks.function(sh=sh, padd=padd, scope=scope, window=window, **kwargs)

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
        
        p0 = self.pf(self.x)
        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:
                if mask_array[0][row][col][0] == 1:

                    if inspect[0] == row and inspect[1] == col:
                        pass
                    else:
                        row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                        row_idx, col_idx = row_idx + row, col_idx + col

                        temp, temp2 = perturbation.function(x=x_copy, row=row_idx, col=col_idx, **kwargs)

                        if temp2 is False:
                            val = self.pf(temp) - p0
                        else:
                            val = (self.pf(temp2) + self.pf(temp)) / 2

                        if absolute:
                            val = abs(val)

                        if np.shape(val):
                            d_temp[0, row:row+window[0], col:col+window[1], 0] = val[evolution]
                        else:
                            d_temp[0, row:row+window[0], col:col+window[1], 0] = val
                else:
                    pass

                col += window[1]
            row += window[0]

        self.syn = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.syn_norm = (d_temp - min_val) / (max_val - min_val)


    def layer_activation(self, layer=0, act_val= 0, p=0.005, window=1, scope=None, padding='center', **kwargs):
        """
        Counts the unit activations in the selected `layer` of a `Keras` model according 
            to change in the feature.
        
        :type layer: int
        :param layer: Position number within the keras model to be analyzed. Use `model.summary()`
            to see exactly the position of the desired layer.

        :type act_val: float
        :param act_val: lower limit from which a unit is considered active, 
            not included (`activation > act_val`). Default is 0 (`relu`).

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
        error_handling(window, scope, padding, False, 0, [1, 2, 3], layer, p)

        # Initial values
        sh = np.array(self.x).shape
        x_copy = copy.deepcopy(self.x)
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))

        scope, window, padding, _ = params_std(self.y, sh, scope, window, padding, None)
        padd = calc_pad(padding, scope, window)
        mask_array = masks.function(sh=sh, padd=padd, scope=scope, window=window, **kwargs)

        # Keras
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)

        p0 = 0
        activations = activation_model.predict(self.x, verbose=0)
        p0 += np.sum(np.maximum(act_val, activations[layer]) > act_val)

        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:

                if mask_array[0][row][col][0] == 1:

                    row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                    row_idx, col_idx = row_idx + row, col_idx + col
                    
                    temp, temp2 = perturbation.function(x=x_copy, row=row_idx, col=col_idx, **kwargs)

                    p1, p2 = 0, 0
                    if temp2 is False:
                        activations = activation_model.predict(temp, verbose=0)
                        p1 += np.sum(np.maximum(act_val, activations[layer]) > act_val)
                        val = p1 - p0
                    else:
                        activations = activation_model.predict(temp, verbose=0)
                        p1 += np.sum(np.maximum(act_val, activations[layer]) > act_val)

                        activations = activation_model.predict(temp2, verbose=0)
                        p2 += np.sum(np.maximum(act_val, activations[layer]) > act_val)

                        val = (p1 + p2 - 2*p0) / 2
                    
                    d_temp[0, row:row+window[0], col:col+window[1], 0] = val

                else:
                    pass

                col += window[1]
            row += window[0]

        self.lac = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.lac_norm = (d_temp - min_val) / (max_val - min_val)
    

    def unit_activation(self, layer=0, act_val=0, p=1, count='count', window=1,
                            scope=None, padding='center', **kwargs):
        """
        Counts the number of activations for each of the units according to the change in the features.
        Alternatevly, it can calulate the `average` or `accumulated` values using the parameter `count`.
        
        :type layer: int
        :param layer: Position number within the keras model to be analyzed. Use `model.summary()`
            to see exactly the position of the desired layer.

        :type act_val: float
        :param act_val: lower limit from which a unit is considered active, 
            not included (`activation > act_val`). The maximum value between `act_val` and
            the quantile value for `p` is used as threshold as `max(act_val, quantile)`. 
            Default is 0 (`relu`).

        :type p: float
        :param p: quantile value for activations, that is, the `p` percentage highest values. 
            A good first approach is to it it to 0.005 (0.5%). The maximum value between `act_val`
            and the quantile value for `p` is used as threshold as `max(act_val, quantile)`. 
            Default is 1.

        :type count: string
        :param count: Method of calculation. Normally you would't whant to change this as other
            options lack interpretability, but they are there if you want. Default is `count`.

        :type window: int, tupple
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
        error_handling(window, scope, padding, False, 0, [1, 2, 3], layer, p)

        # Initial values
        sh = np.array(self.x).shape
        x_copy = copy.deepcopy(self.x)

        scope, window, padding, _ = params_std(self.y, sh, scope, window, padding, None)
        padd = calc_pad(padding, scope, window)
        mask_array = masks.function(sh=sh, padd=padd, scope=scope, window=window, **kwargs)

        # Keras
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)

        p0 = calc_p_uac(layer, activation_model, self.x, act_val, p)
        # ids = [i for i in range(len(p0))] # name of the units as index value
        d_temp = [0 for _ in range(len(p0))]

        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:

                if mask_array[0][row][col][0] == 1:

                    row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                    row_idx, col_idx = row_idx + row, col_idx + col
                    
                    temp, temp2 = perturbation.function(x=x_copy, row=row_idx, col=col_idx, **kwargs)

                    p1, p2 = 0, 0
                    if temp2 is False:
                        p1 = calc_p_uac(layer, activation_model, temp, act_val, p)
                        val = p1 - p0
                        val_count = np.where(val > act_val, 1, 0)

                    else:
                        p1 = calc_p_uac(layer, activation_model, temp, act_val, p)
                        p2 = calc_p_uac(layer, activation_model, temp2, act_val, p)
                        val = (p1 + p2 - 2*p0) / 2
                        val_count = np.where(val > act_val, 1, 0)
                    
                    match count:
                        case 'count':
                            d_temp += val_count
                        case 'average':
                            val_count[val_count <= 0] = 1
                            d_temp += val/val_count
                        case 'accumulate':
                            d_temp += val
                        case _:
                            raise ValueError("Expected string value for `count` to be either `count`,"
                                              "`average`, or `accumulate`. Got instead: %s" % str(count))
                else:
                    pass

                col += window[1]
            row += window[0]

        self.uac = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.uac_norm = (d_temp - min_val) / (max_val - min_val)


###
def unit_to_feature(self, layer=0, act_val=0, p=1, count='count', window=1,
                            scope=None, padding='center', **kwargs):
        """
        ...
        
        :type layer: int
        :param layer: Position number within the keras model to be analyzed. Use `model.summary()`
            to see exactly the position of the desired layer.

        :type act_val: float
        :param act_val: lower limit from which a unit is considered active, 
            not included (`activation > act_val`). The maximum value between `act_val` and
            the quantile value for `p` is used as threshold as `max(act_val, quantile)`. 
            Default is 0 (`relu`).

        :type p: float
        :param p: quantile value for activations, that is, the `p` percentage highest values. 
            A good first approach is to it it to 0.005 (0.5%). The maximum value between `act_val`
            and the quantile value for `p` is used as threshold as `max(act_val, quantile)`. 
            Default is 1.

        :type count: string
        :param count: Method of calculation. Normally you would't whant to change this as other
            options lack interpretability, but they are there if you want. Default is `count`.

        :type window: int, tupple
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
        error_handling(window, scope, padding, False, 0, [1, 2, 3], layer, p)

        # Initial values
        sh = np.array(self.x).shape
        x_copy = copy.deepcopy(self.x)

        scope, window, padding, _ = params_std(self.y, sh, scope, window, padding, None)
        padd = calc_pad(padding, scope, window)
        mask_array = masks.function(sh=sh, padd=padd, scope=scope, window=window, **kwargs)

        # Keras
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)

        p0 = calc_p_uac(layer, activation_model, self.x, act_val, p)
        # ids = [i for i in range(len(p0))] # name of the units as index value
        d_temp = [0 for _ in range(len(p0))]

        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:

                if mask_array[0][row][col][0] == 1:

                    row_idx, col_idx = np.meshgrid(range(window[0]), range(window[1]), indexing='ij')
                    row_idx, col_idx = row_idx + row, col_idx + col
                    
                    temp, temp2 = perturbation.function(x=x_copy, row=row_idx, col=col_idx, **kwargs)

                    p1, p2 = 0, 0
                    if temp2 is False:
                        p1 = calc_p_uac(layer, activation_model, temp, act_val, p)
                        val = p1 - p0
                        val_count = np.where(val > act_val, 1, 0)

                    else:
                        p1 = calc_p_uac(layer, activation_model, temp, act_val, p)
                        p2 = calc_p_uac(layer, activation_model, temp2, act_val, p)
                        val = (p1 + p2 - 2*p0) / 2
                        val_count = np.where(val > act_val, 1, 0)
                    
                    """only in Py 3.10 and above :("""
                    # match count:
                    #     case 'count':
                    #         d_temp += val_count
                    #     case 'average':
                    #         val_count[val_count <= 0] = 1
                    #         d_temp += val/val_count
                    #     case 'accumulate':
                    #         d_temp += val
                    #     case _:
                    #         raise ValueError("Expected string value for `count` to be either `count`,"
                    #                           "`average`, or `accumulate`. Got instead: %s" % str(count))
                    if count == 'count':
                        d_temp += val_count
                    elif count == 'average':
                        val_count[val_count <= 0] = 1
                        d_temp += val/val_count
                    elif count == 'accumulate':
                        d_temp += val
                    else:
                        raise ValueError("Expected string value for `count` to be either `count`,"
                                        "`average`, or `accumulate`. Got instead: %s" % str(count))
                else:
                    pass

                col += window[1]
            row += window[0]

        self.uac = d_temp
        
        max_val, min_val = d_temp.max(), d_temp.min()
        self.uac_norm = (d_temp - min_val) / (max_val - min_val)
###


    def preview(self, window=1, scope=None, padding='center', axis=None, show_data=True, 
                title='Preview', xlabel='Feature', ylabel='Intensity', xticks=None, 
                yticks=[], cmap='Greens', font_size=15, figsize=(14, 4), bold=False, **kwargs):
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
        error_handling(window, scope, padding, False, 0, [1, 2, 3], None, None)

        # Initial values
        image = [] # this will be the preview image
        sh = np.array(self.x).shape
        d_temp = np.zeros((sh[0], sh[1], sh[2], sh[3]))
        countour = np.zeros((sh[0], sh[1], sh[2], sh[3]))
                
        scope, window, padding, _ = params_std(self.y, sh, scope, window, padding, evolution=None)
        padd = calc_pad(padding, scope, window)
        mask_array = masks.function(sh=sh, padd=padd, scope=scope, window=window, **kwargs)

        row = padd[0][0] + scope[0][0]
        while row <= scope[0][1] - padd[0][1] - window[0]:
            col = padd[1][0] + scope[1][0]
            while col <= scope[1][1] - padd[1][1] - window[1]:

                if mask_array[0][row][col][0] == 1:
                    countour[0, row, col, 0] = 1
                    d_temp[0, row:row+window[0], col:col+window[1], 0] = 1

                    if sh[1] > 1:
                        for i in range(sh[1]):
                            countour[0, i, col, 0] = 1
                        for i in range(sh[2]):
                            countour[0, row, i, 0] = 1
                else:
                    pass

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
        # plt.colorbar()
        
        plt.title(title) 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yticks(yticks)

        if xticks:
            plt.xticks(axis, xticks, rotation='vertical')
        
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


def plot(feature, image, axis=None, show_data=True, title='Importance', 
        xlabel='Feature', ylabel='Intensity', xticks=None, yticks=[], cmap='Greens',
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


def calc_pad(padding, scope, window):
    """
    Calculates padding for the given type.

    :type padding: str
    :param padding: Type of padding. Can be 'center', 'left', or 'right'.

    :rtype: list
    :returns: the padding as a list of two integers representing the padding for 
        the left and right sides.
    """
    pad = [[0, 0], [0, 0]]

    for i in range(2):
        comp = int((scope[i][1]-scope[i][0])%window[i])
        if comp > 0:
            if padding[i] == 'center':
                if comp % 2 == 0:  # even number
                    pad[i] = [int(comp / 2), int(comp / 2)]
                else:  # if odd number, left gets the +1
                    pad[i] = [int(np.ceil(comp / 2)), int(np.floor(comp / 2))]
            elif padding[i] == 'left':
                pad[i] = [comp, 0]
            elif padding[i] == 'right':
                pad[i] = [0, comp]
            else:
                raise ValueError(f"Invalid padding type '{padding}'. Valid types are 'center', 'left', and 'right'.")
    
    return pad


def params_std(y, sh, scope, window, padding, evolution):
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
        evolution = int(y)
    
    return scope, window, padding, evolution


def error_handling(window, scope, padding, absolute, inspect, steps, layer, p):
    """
    Handles erros for the main parameters of the different functions.
    """

    if p is None:
        pass
    if not (0 <= p <= 1):
        raise ValueError(f"Expected value for `p` is `0 <= p <= 1`. Got instead: %s" % str(p))

    if layer is None:
        pass
    if (not isinstance(layer, int)) or layer < 0:
        raise ValueError(f"Expected value for `layer` is an integer greater than 0. Got instead: %s" % str(layer))

    if len(np.array(window).shape) > 0:
        if window[0] <= 0 or window[1] <= 0:
            raise ValueError(f"Value for window, or its components, must be greater \
                            than 0. Got instead: %s" % str(window))
    elif window <= 0:
        raise ValueError(f"Value for window, or its components, must be greater than 0.\
                        Got instead: %s" % str(window))
    
    if scope is None:
        pass
    elif len(np.array(scope).shape) == 1:
        if scope[0][0] <= 0 or scope[0][1] <= 0 or scope[1][0] <= 0 or scope[1][1] <= 0:
            raise ValueError("Expected value for the components of scope must be greater than 0."
                                "Got instead: %s" % str(scope))
    elif scope[0] <= 0 or scope[1] <= 0:
        raise ValueError("Expected value for the components of scope must be greater than 0."
                            "Got instead: %s" % str(scope))

    if padding not in ['center', 'left', 'right']:
        raise ValueError("Expected value for padding to be either center, left, or right."
                            "Got instead: %s" % str(padding))

    if not isinstance(absolute, bool):
        raise ValueError("Expected value for absolute is boolean: True or False."
                            "Got instead: %s" % str(absolute))
    
    if len(np.array(inspect).shape) > 0:
        if inspect[0] <= 0 or inspect[1] <= 0:
            raise ValueError("Value for inspect, or its components, must be greater"
                                "than 0. Got instead: %s" % str(inspect))
        
    elif (not isinstance(inspect, int)) or inspect < 0:
        raise ValueError("Expected value for inspect is an integer greater than 0." 
                            "Got instead: %s" % str(inspect))

    if not isinstance(steps, list):
        raise ValueError("Expected value for inspect is a list. Got instead: %s" % str(steps))


def calc_p_uac(layer, activation_model, temp, act_val, p):
    activations = activation_model.predict(temp, verbose=0)
    activations = activations[layer][0]
    activations = np.array(activations).flatten()
    quantile = np.quantile(activations, 1-p)
    activations[activations <= max(quantile, act_val)] = 0

    # val = np.array([np.maximum(max(quantile, act_val), i) for i in activations])

    # val = np.array([np.maximum(max(quantile, act_val), i) for i in activations])

    return activations