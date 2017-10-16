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
#import astra
import numpy
import warnings

# Own modules:
from analysis import process
from analysis import postprocess
from analysis import display
from analysis import analyse

from data import io        
from data import data_blocks
from data import data_blocks_ssd
from meta import proj_meta
from meta import vol_meta

from reconstruction import reconstruct

# System modules:
#import gc

# **************************************************************
#           VOLUME class
# **************************************************************
class volume(object):
    """
    Container for the reconstructed volume data.
    """
    # Reconstructed data:
    data = None
    
    # Meta data (geometry, history etc.):
    meta = None
    
    # Sub-classes:
    io = None
    process = None
    display = None
    analyse = None
    
    # ASTRA related:
    _vol_geom = None
    
    def __init__(self, size, size_mm):
        
        self.data = data_blocks(block_sizeGB = 1)
        self.data.dim = 0
        
        self.meta = vol_meta(self, size, size_mm)
        
        self.io = io(self)
        self.process = postprocess(self)
        self.display = display(self)
        self.analyse = analyse(self)
        
# **************************************************************
#           PROJECTIONS class
# **************************************************************

class projections(object):
    """
    Container for the projection data.
    """
    # Main container for the data:
    data = None 
    
    # Corresponding meta data (geometry, history):
    meta = None
    
    # Some data handling routines:
    io = None
    process = None
    display = None
    analyse = None
    
    _ref  = []
    _dark = []
    
    def __init__(self):
        
        self.data = data_blocks(block_sizeGB = 1)
        self.meta = proj_meta(self)
        
        self.io = io(self)
        self.process = process(self)
        self.display = display(self)
        self.analyse = analyse(self)
        
    def switch_to_ram(self, keep_data = False, block_sizeGB = 1):
        """
        Switches data to a RAM based array.
        """
        if isinstance(self.data, data_blocks):
            print('Warning! Can`t switch to RAM based data, it already is RAM based!')
            
            return 
            
        if keep_data:
            # First copy the data:
            self.data = data_blocks(array = self.data.total, block_sizeGB = block_sizeGB, dtype='float32')
            
        else:
            # Create new:
            self.data = data_blocks(block_sizeGB = block_sizeGB, dtype='float32')
    
    def switch_to_ssd(self, keep_data = False, block_sizeGB = 1, swap_path = '/export/scratch3/kostenko/Fast_Data/swap'):
        """
        Switches data to an SSD based array.
        """
        
        if isinstance(self.data, data_blocks_ssd):
            print('Warning! Can`t switch to SSD based data, it already is SSD based!')
            
            return 
            
        if keep_data:
            # First copy the data:
            self.data = data_blocks_ssd(array = self.data.total, block_sizeGB = block_sizeGB, dtype='float32', swap_path = swap_path)
            
        else:
            # Create new:
            self.data = data_blocks_ssd(block_sizeGB = block_sizeGB, dtype='float32', swap_path = swap_path)
    
    def __del__(self):
        
        # Make sure the data is killed: 
        self.data = None
            
    def message(self, msg):
        '''
        Send a message to IPython console.
        '''
        print(msg)

    def error(self, msg):
        '''
        Throw an error:
        '''
        self.meta.history.add_record('error', msg)
        raise ValueError(msg)

    def warning(self, msg):
        '''
        Throw a warning. In their face!
        '''
        self.meta.history.add_record('warning', msg)
        warnings.warn(msg)
    
    def get_ref(self, proj_num = 0):
        '''
        Returns a reference image. Interpolated if several reference images are available.
        '''
        
        # Return reference image for the current projection:
        if self._ref.ndim > 2:
          
            if self.data is None:
                self._parent.warning('No raw data available. We don`t know how many projections there are in order to interpolate the reference image properly. Read raw data first.')
                dsz = self._ref.shape[1]

            else:
                dsz = self.data.shape[1]
                
            # Several flat field images are available:
            ref = self._ref
          
            sz = ref.shape
            
            proj_index = numpy.linspace(0, sz[1]-1, dsz)
            
            a = proj_index[proj_num]
            fract = a - numpy.floor(a)            
            a = int(numpy.floor(a))            
            
            if a < (dsz-1):
                b = int(numpy.ceil(proj_index[proj_num]))
            else:
                b = a
                
            return self._ref[:, a, :] * (1 - fract) + self._ref[:, b, :] * fract
          
        else:
            
           # One flat field image is available:
           return self._ref  

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

    reconstruct = []#reconstruct()
    
    def __init__(self):
        
        reconstruct = reconstruct()
        
        print('Tomobox is created.')
        
    def load_data(self, path, add_new = False):
        
        #data = 
        self.data.append(data)
    
    
    