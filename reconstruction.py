#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains tomographic reconstruction routines based on ASTRA.
"""
import astra
import data
import misc

import astra.experimental as asex
import numpy
import sys

import scipy.ndimage.interpolation as interp

# **************************************************************
#           RECONSTRUCT class
# **************************************************************
class reconstruct(object):
    '''
    Class that will help a tired wanderer to create beautiful images of kindersuprises. 
    '''
    
    projections = []
    volumes = []
    
    options = {'swap': False, 'constraints': None, 'poisson_stat': False, 'ramp': 0, 'weight': False}

    swap_path = '/export/scratch3/kostenko/Fast_Data/swap'
        
    def __init__(self, projections, volume):
        
        # Remember the projection data that we will use to reconstruct:
        self.add_projections(projections)
        
        # This is your volume:
        self.volume = volume    
            
    def add_projections(self, projections):
        
        if isinstance(projections, list):
            self.projections.extend(projections)
        else:
            self.projections.append(projections)
                

    def _backproject_block(self, proj_data, proj_geom, vol_data, vol_geom, algorithm):
        '''
        Backproject a single block of data
        '''
        
        # TODO:
        # Do we need to introduce ShortScan parameter?    
        # At the moment we will ignore MinConstraint / MaxConstraint    
                
        try:
            sin_id = astra.data3d.link('-sino', proj_geom, proj_data)
            
            vol_id = astra.data3d.link('-vol', vol_geom, vol_data)

            projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
            
            if algorithm == 'BP3D_CUDA':
                asex.accumulate_BP(projector_id, vol_id, sin_id)
            else:
                asex.accumulate_FDK(projector_id, vol_id, sin_id)
          
        except:
            print("ASTRA error:", sys.exc_info())
            
        finally:
            astra.data3d.delete(sin_id)
            astra.data3d.delete(vol_id)
            astra.projector.delete(projector_id)
              
        return vol_data  

    def backproject(self, algorithm = 'BP3D_CUDA', multiplier = 1):
        '''
        ASTRA backprojector. No filtering by default.
        '''
        
        # Volume geometry and projection geometry for ASTRA:
        vol_geom = self.volume.meta.geometry.get_vol_geom()
        
        # This will only work with RAM volume data! Otherwice, need to set data to 'total'.
        if isinstance(self.volume.data, data.data_blocks_swap):
            raise ValueError('Backprojection doesn`t support SSD data blocks for volumes!')
            
        # Pointer to the total volume:    
        vol_data = self.volume.data.total
        
        if vol_data.size == 0:
            raise ValueError('Volume data array is empty. Cannot backproject into an empty array.')
        
        # Loop over different projection stacks:
        for proj in self.projections:        
            
            # Loop over blocks of data to save RAM:
            for block in proj.data:
                
                # ASTRA projection geometry for the current block:
                proj_geom = proj.meta.geometry.get_proj_geom(blocks = True)
                
                # Backprojection:
                if multiplier != 1:    
                    self._backproject_block(multiplier * block, proj_geom, vol_data, vol_geom, algorithm) 
                    
                else:
                    self._backproject_block(block, proj_geom, vol_data, vol_geom, algorithm) 
                
            # Apply constraints:
            constr = self.options['constraints']
            if constr is not None:
                numpy.clip(vol_data, a_min = constr[0], a_max = constr[1], out = vol_data)

    def _forwardproject_block(self, proj_data, proj_geom, vol_data, vol_geom):
        '''
        Forwardproject a single block of data
        '''
            
        try:  
            
          sin_id = astra.data3d.link('-sino', proj_geom, proj_data)  
          vol_id = astra.data3d.link('-vol', vol_geom, vol_data)         
          
          projector_id = astra.create_projector('cone', proj_geom, vol_geom)
          
          asex.accumulate_FP(projector_id, vol_id, sin_id) 
          
        finally:
          astra.data3d.delete(sin_id)
          astra.data3d.delete(vol_id)
          astra.projector.delete(projector_id)
          
        return proj_data                  
                
    def forwardproject(self, multiplier = 1):
        '''
        ASTRA forwardprojector. 
        '''
        
        # Volume geometry and projection geometry for ASTRA:
        vol_geom = self.volume.meta.geometry.get_vol_geom()
        
        # Loop over different projection stacks:
        for proj in self.projections:
            
            # This will only work with RAM volume data! Otherwise, need to use 'total' setter.
            if isinstance(proj.data, data.data_blocks_swap):
                raise ValueError('Forwardprojection doesn`t support swap data blocks for projections!')
            
            proj_data = proj.data.total
            
            if proj_data.size == 0:
              raise ValueError('Projection data array is empty. Cannot backproject into an empty array.')
        
            # ASTRA projection geometry:
            proj_geom = proj.meta.get_proj_geom()
            
            # Loop over blocks of data to save RAM:
            for block in self.volume.data:
                
                # Generate geometry for the current block:
                vol_geom = self.volume.meta.geometry.get_vol_geom(blocks = True)
                
                if multiplier != 1:    
                    self._forwardproject_block(proj_data, proj_geom, multiplier * block, vol_geom)    
                    
                else:
                    self._forwardproject_block(proj_data, proj_geom, block, vol_geom)    
        
    def FDK(self):
        '''
        The method of methods.
        '''
        # Switch to a different data storage if needed:
        self.volume.switch_to_ram(keep_data = True)
        
        # Run the reconstruction:
        self.backproject(algorithm='FDK_CUDA')

        # Update history:    
        self.volume.meta.history.add_record('Reconstruction generated using FDK.', [])
        
    def _volume_to_swap(self):
        '''
        Put volume to swap and projections to RAM. Do it before forwardprojection
        '''
        
        if self.options['swap']:
            self.volume.switch_to_swap(keep_data = True, swap_path = self.swap_path)
        
            for proj in self.projections:
                proj.switch_to_ram(keep_data = True)
            
    def _projections_to_swap(self):
        '''
        Put projections to swap and volume to RAM. Do it before backprojection
        '''
        
        if self.options['swap']:
            for proj in self.projections:
                proj.switch_to_swap(keep_data = True, swap_path = self.swap_path)
                
            self.volume.switch_to_ram(keep_data = True)
            
    def _compute_forward_weight(self):
        '''
        Compute weights applied to the forward projection. FP(BP(ONES))
        '''        
        
        
    def SIRT(self, iterations = 3):
        '''
        Simultaneous Iterative Reconstruction Technique... also known as James
        '''        
        
        # Compute weights:
        print('Computing weights...')
        
        # Create temporary volume of ones...
        # Create temporary projections...
        # Back
        # Forward
        # Max projections
        
        # Weight to be on the safe side:
        w = self.volume.data.shape.max() 
        
        for ii in range(iterations):
            # Switch data to swap if needed:
            self._volume_to_swap()
            self.forwardproject(multiplier = -1)
            
            self._projections_to_swap()
            self.backproject(multiplier = 1 / w)
            
            misc.progress_bar((ii + 1) / iterations)
        
    def CPLS(self):
        '''
        Chambolle-Pock Least Squares
        '''
        pass
        
    def EM(self):
        '''
        Expectation maximization
        '''
        pass
        
    def FISTA(self):
        '''
        
        '''
        pass
    
    def stitch_tiles(self, swap_path):
        '''
        Stich several projection tiles into one projection.
        
        '''
        
        print('Stitching tiles...')
        
        pix = self.projections[0].meta.geometry.det_pixel
        theta_n = self.projections[0].meta.geometry.theta_n
        
        min_x, min_y = numpy.inf
        max_x, max_y = -numpy.inf
        
        # Find out the size required for the final dataset
        for proj in self.projections:
            xx, yy = proj.meta.geometry.get_pixel_coords()
            
            min_x = min((min_x, min(xx)))
            min_y = min((min_y, min(yy)))
            max_x = max((max_x, max(xx)))
            max_y = max((max_y, max(yy)))
            
        # Big slice:
        total_shape = [(max_y - min_y) / pix[1], theta_n, (max_x - min_x) / pix[0]]     
        total_centre = [(max_y + min_y) / 2, (max_x - min_x) / 2]

        print('Total dataset size is', total_shape)
                
        # Initialize a large projection array:
        new_data = data.data_blocks_swap(shape = total_shape, block_sizeGB = 1, dim = 1, swap_path = swap_path)            
        
        # Interpolate slice by slice:
        for ii in range(new_data.length): 
            
            total_slice = new_data.empty_slice()
            total_slice_size = [(max_y - min_y), (max_x - min_x)]
            
            for proj in self.projections:
                
                # Detector size in mm:
                det_size = proj.meta.geometry.det_size
                
                # Get a tile image:
                img = proj.data.get_slice(ii)
                
                # Difference between total size and current tile size:
                pad_x = total_shape[1] - img.shape[1]
                pad_y = total_shape[0] - img.shape[0]
                img = numpy.pad(img, ((0, pad_y), (0, pad_x)))
                
                # Offset from the left top corner:
                offset = (proj.meta.geometry.det_trans - total_centre) + total_slice_size / 2 - det_size / 2
                
                print('shifting by offset:', offset)
                
                total_slice += interp.shift(img, offset)
                
            new_data[ii] = total_slice 

            misc.progress_bar(ii / new_data.length)
            
        return new_data
        
# TODO: random access
    
