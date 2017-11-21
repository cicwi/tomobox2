#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

Just usefull stuff...

"""

import time

class subclass(object):
    def __init__(self, parent):
        self._parent = parent

# Use global time variable to measure time needed to compute stuff:        
glob_time = 0

def progress_bar(progress):
    """
    Plot progress in pseudographics:
    """
    global glob_time 
    
    
    if glob_time == 0:
        glob_time = time.time()
    
    print('\r', end = " ")
    
    bar_length = 40
    if progress >= 1:
        
        # Repoort on time:
        txt = 'Done in %u sec!' % (time.time() - glob_time)
        glob_time = 0
        
        for ii in range(bar_length):
            txt = txt + ' '
            
        print(txt) 

    else:
        # Build a progress bar:
        txt = '\u2595'
        
        for ii in range(bar_length):
            if (ii / bar_length) <= progress:
                txt = txt + '\u2588'
            else:
                txt = txt + '\u2592'
                
        txt = txt + '\u258F'        
        
        print(txt, end = " ") 
        
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