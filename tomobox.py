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
#import gc

# Own modules:
from analysis import preprocess
from analysis import postprocess
from analysis import display
from analysis import analyse

from data import io        
from data import data_array
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
    
    def __init__(self, block_sizeGB = 1, swap = False):
        
        # Default RAM based data array:
        self.data = data_array(dtype = 'float32', block_sizeGB = block_sizeGB, swap = swap)        
        
        # Common classes for the cvolume and for the projections:
        self.io = io(self)
        self.display = display(self)
        self.analyse = analyse(self)
        
        # Specific to the projection data or the volume data:
        self.process = None
        self.meta = None
    
    def __del__(self):
        
        # Kill the data! Free the memory.
        self.release()
        
    def release(self): 
        '''
        Release resources.
        '''
        self.data.release()
        
# **************************************************************
#           VOLUME class
# **************************************************************
class volume(tomographic_data):
    """
    Container for the reconstructed volume data.
    """
      
    def __init__(self, array = None, shape = None, img_pixel = [1,1,1], block_sizeGB = 1, swap = False):
        """
        Initialize.
        """
        # Initializa parent class:
        tomographic_data.__init__(self, block_sizeGB, swap)
        
        if array is not None:
            self.data.total = array
            
        elif shape is not None:
            self.data.total = numpy.zeros(shape, dtype = numpy.float32)
        
        #else:
        #    raise ValueError('Come on, man! You need to specify at least the array size.')

        # Volume-specific stuff: 
        self.meta = vol_meta(self)
        self.process = postprocess(self)
            
        # Set the correct main axis for the data object:
        self.data.dim = 0
        self.meta.geometry.img_pixel = img_pixel    
                 
    def initialize_volume(self, size, img_pixel):
        '''
        Initialize a dataset of a given size
        '''        
        
        self.data.total = numpy.zeros(size, dtype = numpy.float32)
        self.meta.geometry.img_pixel = img_pixel
        
    def copy(self, swap = False):
        
        array = self.data.total 
        img_pixel = self.meta.geometry.img_pixel
        block_sizeGB = self.data._block_sizeGB

        vol = volume(array = array, img_pixel = img_pixel, block_sizeGB = block_sizeGB, swap = swap)
        vol.data.dim = 0
        
        return vol
     
# **************************************************************
#           PROJECTIONS class
# **************************************************************

class projections(tomographic_data):
    """
    Container for the projection data.
    """
    
    def __init__(self, block_sizeGB = 1, swap = False):
        
        # Initializa parent class:
        tomographic_data.__init__(self, block_sizeGB, swap)
           
        # Projections-specific fields:    
        self._ref  = []
        self._dark = []
        
        self.meta = proj_meta(self)
        self.process = preprocess(self)
        
        # Sinograms should have dim = 1 as a main axis:
        self.data.dim = 1
                
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
    
    
    