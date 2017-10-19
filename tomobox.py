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
import gc

# Own modules:
from analysis import process
from analysis import postprocess
from analysis import display
from analysis import analyse

from data import io        
from data import data_blocks
from data import data_blocks_swap
from meta import proj_meta
from meta import vol_meta

from reconstruction import reconstruct


# **************************************************************
#           TOMOGRAPHIC_DATA class
# **************************************************************
class tomographic_data(object):
    """
    This is a data container class that will be inherited by volume and projections classes.
    """
    
    # Data:
    data = None    
    meta = None
    
    # Sub-classes:
    io = None
    process = None
    display = None
    analyse = None
    
    def __init__(self):
        # Default RAM based data array:
        self.data = data_blocks(block_sizeGB = 1)        
        
        # Common classes for the cvolume and for the projections:
        self.io = io(self)
        self.display = display(self)
        self.analyse = analyse(self)
    
    def switch_to_ram(self, keep_data = False, block_sizeGB = None):
        """
        Switches data to a RAM based array.
        """
        
        if not isinstance(self.data, data_blocks_swap):
            print('I am already in RAM memory!')
            
            return 
            
        if block_sizeGB is None:
            block_sizeGB = self.data._block_sizeGB

        if keep_data:
            
            # First copy the data:
            new_data = data_blocks(array = self.data.total, block_sizeGB = block_sizeGB)
        else:
            # Create new:
            new_data = data_blocks(block_sizeGB = block_sizeGB)
            
        self.data = new_data  
        
        # Clean up!
        gc.collect()

        print('Switched to data_blocks_ram')
    
    def switch_to_swap(self, keep_data = False, block_sizeGB = None, swap_path = '/export/scratch3/kostenko/Fast_Data/swap'):
        """
        Switches data to an SSD based array.
        """
        
        if isinstance(self.data, data_blocks_swap):
            print('We are already swappin!')
            return 
            
        if block_sizeGB is None:
            block_sizeGB = self.data._block_sizeGB    
            
        if keep_data:
            # First copy the data:
            new_data = data_blocks_swap(array = self.data.total, block_sizeGB = block_sizeGB, dtype='float32', swap_path = swap_path)
            
        else:
            # Create new:
            new_data = data_blocks_swap(block_sizeGB = block_sizeGB, dtype='float32', swap_path = swap_path)   
            
        self.data = new_data  
        
        # Clean up!
        gc.collect()
        
        print('Switched to data_blocks_swap')

# **************************************************************
#           VOLUME class
# **************************************************************
class volume(tomographic_data):
    """
    Container for the reconstructed volume data.
    """
      
    def __init__(self, size = [0,0,0], img_pixel = [0,0,0]):
        
        # Initializa parent class:
        tomographic_data.__init__(self)
        
        # Set the correct main axis for the data object:
        self.data.dim = 0
        
        self.meta = vol_meta(self)
        self.process = postprocess(self)
        
        self.initialize_volume(size, img_pixel)
        
    def initialize_volume(self, size, img_pixel):
        '''
        Initialize a dataset of a given size
        '''        
        
        self.data.total = numpy.zeros(size, dtype = numpy.float32)
        self.meta.geometry.img_pixel = img_pixel
        
# **************************************************************
#           PROJECTIONS class
# **************************************************************

class projections(tomographic_data):
    """
    Container for the projection data.
    """
    # Projections-specific fields:    
    _ref  = []
    _dark = []
    
    def __init__(self):
        
        # Initializa parent class:
        tomographic_data.__init__(self)
        
        # Sinograms should have dim = 1 as a main axis:
        self.data.dim = 1
        self.meta = proj_meta(self)
        
        # Processing for sinograms:
        self.process = process(self)
        
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
#           TOMOBOX class
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
    
    
    