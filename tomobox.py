#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains the Great Mighty Tomobox.
"""

# **************************************************************
#           Imports
# **************************************************************

# Numerical modules: 
import astra
import numpy

# Own modules:
from data import data        
from reconstruction import reconstruct    

# System modules:
import gc

# **************************************************************
#           SINOGRAM class
# **************************************************************

class tomobox(object):
    """ 
    Tomobox allows to load, process and reconstruct tomographic data from
    one or multiple sources simultaneously. 
    
    ASTRA is used as the core engine for projection / backprojection.
    """
    
    data = []
    volume = []

    reconstruct = reconstruct()
    
    def __init__(self):
        
        reconstruct = reconstruct()
        
        print('Tomobox is created.')
        
    def load_data(self, path, add_new = False):
        
        data = 
        self.data.append(data)
    
    
    