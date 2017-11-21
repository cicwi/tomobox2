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
class _tomographic_data(object):
    """
    This is a data container class that will be inherited by volume and projections classes.
    """
    
    def __init__(self, block_sizeGB = 1, swap = False, pool_dtype = 'float32'):
        
        # Default RAM based data array:
        self.data = data_array(dtype = 'float32', block_sizeGB = block_sizeGB, swap = swap, pool_dtype = pool_dtype)        
        
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
class volume(_tomographic_data):
    """
    Container for the reconstructed volume data.
    """
      
    def __init__(self, projections = None, array = None, shape = None, img_pixel = None, block_sizeGB = 1, swap = False, pool_dtype = 'float32'):
        """
        Initialize.
        """
        # Initializa parent class:
        _tomographic_data.__init__(self, block_sizeGB, swap, pool_dtype = pool_dtype)
        
        # Set the correct main axis for the data object:
        self.data.dim = 0
        
        # Volume-specific stuff: 
        self.meta = vol_meta(self)
        self.process = postprocess(self)
        
        # If projections are defined - use their properties to initialize the volume:
        if projections is not None:
            sz = projections.data.shape[0]
            sx = projections.data.shape[2]

            img_pixel = projections.meta.geometry.img_pixel
            shape = [sz, sx, sx]
        
        if img_pixel is not None:
            self.meta.geometry.img_pixel = img_pixel    
        
        if array is not None:
            self.data.total = array            
        elif shape is not None:
            self.data.total = numpy.zeros(shape, dtype = numpy.float32)
        
                 
    def initialize(self, shape, img_pixel):
        '''
        Initialize a dataset of a given size
        '''        
        self.data.total = numpy.zeros(shape, dtype = self.data.dtype)
        self.meta.geometry.img_pixel = img_pixel
        
    def initialize_from_projections(self, projections):
        '''
        Initialize volume using parameters of the projection data.
        '''
        sz = projections.data.shape[0]
        sx = projections.data.shape[2]

        pix = projections.meta.geometry.img_pixel
        
        self.initialize(shape = [sz, sx, sx], img_pixel = pix)
        
        print('Default volume is generated!')
        
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

class projections(_tomographic_data):
    """
    Container for the projection data.
    """
    
    def __init__(self, block_sizeGB = 1, swap = False, pool_dtype = 'float32'):
        
        # Initializa parent class:
        _tomographic_data.__init__(self, block_sizeGB, swap, pool_dtype)
           
        # Projections-specific fields:    
        self._ref  = []
        self._dark = []
        
        self.meta = proj_meta(self)
        self.process = preprocess(self)
        
        # Sinograms should have dim = 1 as a main axis:
        self.data.dim = 1
        
    def copy(self):
        import copy
        
        #block_sizeGB = self.data._block_sizeGB

        #prj = projections(block_sizeGB = block_sizeGB, swap = swap)
        #prj.meta = prj.meta.copy()
        #prj.data.total = self.data.total        
        prj = copy.deepcopy(self)
                
        return prj
                
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
    
    
    