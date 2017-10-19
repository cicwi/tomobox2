#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains tomographic reconstruction routines based on ASTRA.
"""
import astra
import data

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
    
    # Options:
    constraints = None
        
    
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

    def backproject(self, algorithm = 'BP3D_CUDA', constraints = None):
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
                self._backproject_block(block, proj_geom, vol_data, vol_geom, algorithm) 
                
            # Apply constraints:
            if not constraints is None:
                numpy.clip(vol_data, a_min = constraints[0], a_max = constraints[1], out = vol_data)

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
                
    def forwardproject(self):
        '''
        ASTRA forwardprojector.
        '''
        
        # Volume geometry and projection geometry for ASTRA:
        vol_geom = self.volume.meta.geometry.get_vol_geom()
        
        # Loop over different projection stacks:
        for proj in self.projections:
            
            # This will only work with RAM volume data! Otherwise, need to use 'total' setter.
            if isinstance(proj.data, data.data_blocks_swap):
                raise ValueError('Forwardprojection doesn`t support SSD data blocks for projections!')
            
            proj_data = proj.data.total
            
            if proj_data.size == 0:
              raise ValueError('Projection data array is empty. Cannot backproject into an empty array.')
        
            # ASTRA projection geometry:
            proj_geom = proj.meta.get_proj_geom()
            
            # Loop over blocks of data to save RAM:
            for block in self.volume.data:
                
                # Generate geometry for the current block:
                vol_geom = self.volume.meta.geometry.get_vol_geom(blocks = True)
                
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
        
    def SIRT(self, iterations = 3):
        '''
        Simultaneous Iterative Reconstruction Technique... also known as James
        '''
                
        pass
        """
        for ii_iter in range(iterations):
            
            
            # TODO: SWAP SWAP SWAP!!!!
            
            fwd_proj = self._forwardproject(vol)

            residual = (data - fwd_proj_vols)

            if not self._projection_mask is None:
                residual *= self._projection_mask
            
            vol += self._backproject(residual, algorithm='BP3D_CUDA') / weights #* bwd_weights
            
            if not self._reconstruction_mask is None:
                vol = self._reconstruction_mask * vol
            
            # If relative_constraint is used, force values below relative threshold to 0:
            if relative_constraint != None:
                threshold = vol.max() * relative_constraint
                vol[vol < threshold] = 0

            # Enforce non-negativity or similar:
            if min_constraint != None:
                vol[vol < min_constraint] = min_constraint

            vol_obj.data._data = vol    
            
            print('SIRT_CPU. Iteration %01d' % ii_iter)
            print('Maximum value is:', vol.max())
            
            vol_obj.display.slice(fig_num= 11)
            
        return vol_obj
        """
        
        
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
        
        pix = proj[0].meta.geometry.det_pixel
        theta_n = proj[0].meta.geometry.theta_n
        
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

        print('Total dataset size is', total_slice_shape)
        
        total_slice_shape.append()
        
        # Initialize a large projection array:
        new_data = data_blocks_swap(shape = total_shape, block_sizeGB = 1, swap_path = swap_path)            
        
        # Interpolate slice by slice:
        for ii in range(new_data.length): 
            
            
            total_slice = new_data.empty_slice()
            
            for proj in self.projections:
                
                img = proj.data.get_slice(ii)
                
                pad_x = total_slice.shape[1] - img.shape[1]
                pad_y = total_slice.shape[0] - img.shape[0]
                
                img = numpy.pad(img, ((0, pad_y), (0, pad_x)))
                
                proj.meta.geometry.det_trans[1] - total_centre[1]
                offset = 
                
                interp.shift(img, offset)
        
        for ii in range(new_data.block_number):
            
        
        return new_data
            
    def total_shape
        
# TODO: random access
    
