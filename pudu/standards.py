import numpy as np

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
    """
    This function standardizes the main parameters.

    :type y: float
    :param y: Target. To be used if evolution is not provided.

    :type sh: tuple
    :param sh: Shape of the data.

    :type scope: tuple or None
    :param scope: The scope of the analysis. If None, defaults to full range.

    :type window: tuple or int
    :param window: Size of the window. If a single int is provided, it will be 
        interpreted as (window, window) or (1, window) based on shape.

    :type padding: tuple or int
    :param padding: The amount of padding around the data. If is single 
        value, it is interpreted as (padding, padding).

    :type evolution: int or None
    :param evolution: Evolution to classification groups. If None, 
        defaults to the value of 'y'.

    :rtype: tuple
    :returns: Standardized parameters (scope, window, padding, evolution) as a tuple.
    """
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
    
    padding = calc_pad(padding, scope, window)

    sec_row = (scope[0][1] - scope[0][0] - padding[0][0] - padding[0][1] - window[0]) // window[0] + 1
    sec_col = (scope[1][1] - scope[1][0] - padding[1][0] - padding[1][1] - window[1]) // window[1] + 1
    total = sec_row*sec_col

    return scope, window, padding, evolution, total
