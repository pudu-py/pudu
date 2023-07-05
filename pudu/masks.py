import numpy as np
import random

def function(mask_type='all', sh=None, padd=None, scope=None, window=None, percentage=1,
            qty=None, vector=None,
            mode=None, delta=None, power=None, bias=None, base=None, rr=None, exp=None, 
            th=None, upper=None, lower=None, scale_factor=None, frequency=None, amplitude=None, 
            mean=None, stddev=None, offset=None, constant=None, custom=None):
    
    sec_row = (scope[0][1] - scope[0][0] - padd[0][0] - padd[0][1] - window[0]) // window[0] + 1
    sec_col = (scope[1][1] - scope[1][0] - padd[1][0] - padd[1][1] - window[1]) // window[1] + 1
    
    total = sec_row*sec_col
    
    # for percentage
    to_eval = int(percentage*total)

    # for randompercentage
    to_eval_rand = int(random.randint(0,1)*total)

    # for everyother
    is_eo = 0

    mask = np.zeros((sh[0], sh[1], sh[2], sh[3]))
    section = 1

    row = padd[0][0] + scope[0][0]
    while row <= scope[0][1] - padd[0][1] - window[0]:
        col = padd[1][0] + scope[1][0]
        while col <= scope[1][1] - padd[1][1] - window[1]:
            val = 0
            # print(section)
            if mask_type == 'percentage':
                if section <= to_eval:
                    val = 1

            if mask_type == 'randompercentage':
                if section <= to_eval_rand:
                    val = 1

            elif mask_type == 'quantity':
                if section <= qty:
                    val = 1

            elif mask_type == 'everyother':
                is_eo = (is_eo + 1) % 2
                val = is_eo

            elif mask_type == 'pairs':
                if section%2 == 0:
                    val = 1
            
            elif mask_type == 'odds':
                if section%2 != 0: 
                    val = 1

            elif mask_type == 'random':
                val = random.randint(0,1)
            
            elif mask_type == 'custom':
                val = vector[section]
            
            # elif mask_type == 'circular':
            #     (x−a)2 + (y−b)2 = r**2
            
            # elif mask_type == 'randomwalking':
                
            elif mask_type == 'all':
                val = 1

            else:
                raise ValueError(f"Mask method not recognized, please choose one of from the list or use the\
                        default: %s" % str(['percentage', 'randompercentage', 'quantity', 'everyother', 'pairs',
                                            'odds', 'random', 'custom', 'circular', 'randomwalking', 'all']))

            mask[0, row:row+window[0], col:col+window[1], 0] = val

            section += 1

            col += window[1]
        row += window[0]

    return mask