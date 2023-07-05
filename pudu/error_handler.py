from keras.models import Sequential, Model
import numpy as np

def for_constructor(model, x, y):
    """
    Handles errors for the constructor
    """
    if model is None:
        pass
    elif not isinstance(model, (Sequential, Model)):
        raise ValueError("Expected `model` to be a keras model")

    # if np.array(x).shape[0] > 1 and (not isinstance(y, list)):
    #     raise ValueError("Expected `x` and `y` to have the same length. "
    #                         "Got `x` with length (batch size): %s and `y` is the scalar: %s " 
    #                         % (str(np.array(x).shape[0]), str(y)))
    
    # if isinstance(y, list) and np.array(x).shape[0] != len(y):
    #     raise ValueError("Expected `x` and `y` to have the same length. "
    #                         "Got `x` with length (batch size): %s and `y` with length: %s " 
    #                         % (str(np.array(x).shape[0]), str(len(y))))

    # if len(np.array(x).shape) != 4:
    #     raise ValueError("Expected array to have rank 4 (batch, rows, columns, depth)."
    #                         "Got array with shape: %s" % str(np.array(x).shape))
    
    # if len(np.array(y).shape) != 0:
    #     raise ValueError("Expected integer. Got array with shape: %s" % str(np.array(y).shape))
    x_shape = np.array(x).shape

    if len(x_shape) != 4:
        raise ValueError("Expected array to have rank 4 (batch, rows, columns, depth)."
                            "Got array with shape: %s" % str(x_shape))

    if x_shape[0] == 1:
        if np.array(y).shape != ():
            raise ValueError("Expected `y` to be a scalar when `x` has a batch size of 1."
                            "Got `y` with shape: %s" % str(np.array(y).shape))
    else:
        if not isinstance(y, list) or len(y) != x_shape[0]:
            raise ValueError("Expected `x` and `y` to have the same length when `x` has a batch size more than 1."
                            "Got `x` with length (batch size): %s and `y` with length: %s" 
                            % (str(x_shape[0]), str(len(y))))

def for_params(window, scope, padding, absolute, inspect, steps, layer, p):
    """
    Handles erros for the main parameters of the different functions.
    """

    if p is None:
        pass
    elif not (0 <= p <= 1):
        raise ValueError(f"Expected value for `p` is `0 <= p <= 1`. Got instead: %s" % str(p))

    if layer is None:
        pass
    elif (not isinstance(layer, int)) or layer < 0:
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
    elif len(np.array(scope).shape) > 1:
        if scope[0][0] < 0 or scope[0][1] <= 0 or scope[1][0] < 0 or scope[1][1] <= 0:
            raise ValueError("Expected value for the components of scope must be greater than 0."
                                "Got instead: %s" % str(scope))
    elif scope[0] < 0 or scope[1] <= 0:
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


