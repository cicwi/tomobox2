#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

Just usefull stuff...

"""

class subclass(object):
    def __init__(self, parent):
        self._parent = parent
        

def progress_bar(progress):
    """
    Plot progress in pseudographics:
    """
    
    txt = '\r\u2595'
    
    bar_length = 40
    
    for ii in range(bar_length):
        if (ii / bar_length) <= progress:
            txt = txt + '\u2588'
        else:
            txt = txt + '\u2592'
            
    txt = txt + '\u258F'        
    
    print(txt, end = " ") 
        
    if progress >= 1:
        print(' ')
        
def _set_dim_data(data, dim, key, image):    
    """
    Sets a slice of data along a particular dimension:
    """
    if dim == 0:        
        data[key, :, :] = image

    elif dim == 1:
         data[:, key, :] = image

    else:
        data[:, :, key] = image
        
def _get_dim_data(data, dim, key):
    """
    Gets a slice of data along a particular dimension:
    """
    if dim == 0:        
        return data[key, :, :] 

    elif dim == 1:
        return data[:, key, :]

    else:
        return data[:, :, key]        