#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains tomographic reconstruction routines based on ASTRA.
"""
import astra
import misc

from meta import flex_geometry

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
            
    def __init__(self, projections, volume):
        
        self.projections = []
        self.volumes = []
    
        self.options = {'swap': False, 'constraints': None, 'poisson_stat': False, 'ramp': 0, 'weight': False}

        self.swap_path = '/export/scratch3/kostenko/Fast_Data/swap'
    
        # Remember the projection data that we will use to reconstruct:
        self.add_projections(projections)
        
        # This is your volume:
        self.volume = volume   
        
        # If any of the datasets is in swap need to use swap option!
        if volume.data.is_swap():
            self.options['swap'] = True
            
    def add_projections(self, projections):
        
        if isinstance(projections, list):
            self.projections.extend(projections)
            
            # If any of the datasets is in swap need to use swap option!
            for proj in projections:
                if proj.data.is_swap():
                    self.options['swap'] = True
        else:
            self.projections.append(projections)
            
            # If any of the datasets is in swap need to use swap option!
            if projections.data.is_swap():
                self.options['swap'] = True
                

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
            elif algorithm == 'FDK_CUDA':
                asex.accumulate_FDK(projector_id, vol_id, sin_id)
            else:
                raise ValueError('Unknown ASTRA algorithm type.')
          
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
        It will loop over all registered projection datasets, 
        collect one data block from each at a time and project it.
        '''
        
        # Volume geometry and projection geometry for ASTRA:
        vol_geom = self.volume.meta.geometry.get_vol_geom()
        
        # This will only work with RAM volume data! Otherwice, need to set data to 'total'.
        if self.volume.data.is_swap():
            raise ValueError('Backprojection doesn`t support swap data array for volumes!')
            
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

    def backproject_tiles(self, algorithm = 'BP3D_CUDA', multiplier = 1):
        '''
        ASTRA backprojector. Will stitch tiles before backprojection.
        '''
        
        import matplotlib.pyplot as plt
        
        # Volume geometry and projection geometry for ASTRA:
        vol_geom = self.volume.meta.geometry.get_vol_geom()
        
        # This will only work with RAM volume data! Otherwice, need to set data to 'total'.
        if self.volume.data.is_swap():
            raise ValueError('Backprojection doesn`t support swap data array for volumes!')
            
        # Pointer to the total volume:    
        vol_data = self.volume.data.total
        
        if vol_data.size == 0:
            raise ValueError('Volume data array is empty. Cannot backproject into an empty array.')  
        
        # Assuming all data has the same amount of blocks and projections    
        bn = self.projections[0].data.block_number
            
        for ii in range(bn):
            
            # Blocks from different datasets:
            blocks = []

            # proj_geometries
            proj_geometries = []

            # Loop over different projection stacks:
            for proj in self.projections:  
                
                blocks.append(proj.data[ii])
                proj_geometries.append(proj.meta.geometry.get_proj_geom(blocks = True))
   
            pixel = self.projections[0].meta.geometry.det_pixel

            # Produce one big block:
            print('Stitch a block!')    
            
            big_block = self._stitch_block(blocks, pixel)    
            det_shape = big_block.shape[::2]

            big_geom = flex_geometry.merge_geometries(proj_geometries, det_shape)
            
            plt.imshow(big_block[:,0,:])
            plt.show()
            
            # Backprojection:
            if multiplier != 1:    
                self._backproject_block(multiplier * big_block, big_geom, vol_data, vol_geom, algorithm) 
                
            else:
                self._backproject_block(big_block, big_geom, vol_data, vol_geom, algorithm) 
                                    
            # Apply constraints:
            constr = self.options['constraints']
            if constr is not None:
                numpy.clip(vol_data, a_min = constr[0], a_max = constr[1], out = vol_data)
                
            misc.progress_bar((ii+1) / bn)    
                
    def _forwardproject_block(self, proj_data, proj_geom, vol_data, vol_geom):
        '''
        Forwardproject a single block of data
        '''
            
        try:  
          sin_id = astra.data3d.link('-sino', proj_geom, proj_data)  
          vol_id = astra.data3d.link('-vol', vol_geom, vol_data)         
          
          projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
         
          asex.accumulate_FP(projector_id, vol_id, sin_id) 
          
          print('forward')
          print(proj_geom)   
          print(vol_geom)
          print(proj_data.sum())
          
        except:
          print("ASTRA error:", sys.exc_info())  
          
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
            if proj.data.is_swap():
                raise ValueError('Forwardprojection doesn`t support swap data arrays for projections!')
            
            proj_data = proj.data.total
            
            if proj_data.size == 0:
              raise ValueError('Projection data array is empty. Cannot backproject into an empty array.')
        
            # ASTRA projection geometry:
            proj_geom = proj.meta.geometry.get_proj_geom()
            
            # Loop over blocks of data to save RAM:
            for block in self.volume.data:
                
                # Generate geometry for the current block:
                vol_geom = self.volume.meta.geometry.get_vol_geom(blocks = True)
                
                if multiplier != 1:    
                    self._forwardproject_block(proj_data, proj_geom, multiplier * block, vol_geom)    
                    
                else:
                    self._forwardproject_block(proj_data, proj_geom, block, vol_geom)    
                    
            print('forward total')
            print(proj_data.sum())        
            print(proj.data.total.sum())
        
    def FDK(self):
        '''
        The method of methods.
        '''
        
        print('Switching volume data to RAM')
        
        # Switch to a different data storage if needed:
        self.volume.data.switch_to_ram(keep_data = False)
        
        # Make sure you start form a blank volume
        self.volume.data.zeros()
        
        print('Computing backprojection')
        
        # Run the reconstruction:
        if len(self.projections) > 1:
            self.backproject_tiles(algorithm='FDK_CUDA')
            
        else:
            self.backproject(algorithm='FDK_CUDA')

        # Update history:    
        self.volume.meta.history.add_record('Reconstruction generated using stitched FDK.', [])
        
    def _volume_to_swap(self):
        '''
        Put volume to swap and projections to RAM. Do it before forwardprojection
        '''
        
        if self.options['swap']:
            self.volume.data.switch_to_swap(keep_data = True, swap_path = self.swap_path, swap_name = 'vol_swap')
        
            for proj in self.projections:
                proj.data.switch_to_ram(keep_data = True)
            
    def _projections_to_swap(self):
        '''
        Put projections to swap and volume to RAM. Do it before backprojection
        '''
        
        if self.options['swap']:
            for proj in self.projections:
                proj.data.switch_to_swap(keep_data = True, swap_path = self.swap_path, swap_name = 'proj_swap')
                
            self.volume.data.switch_to_ram(keep_data = True)
            
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
            
            # Preview:
            self.volume.display.slice(dim = 0)
            self.volume.display.slice(dim = 1)
            self.projections[0].display.slice(dim = 1)
            
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
    
    def _stitch_block(self, blocks, pixel):
        '''
        Stich several projection tiles into one projection.
        
        '''        
        # Phisical detector size:
        min_x, min_y = numpy.inf, numpy.inf
        max_x, max_y = -numpy.inf, -numpy.inf
        
        # Find out the size required for the final dataset
        for proj in self.projections:
            xx, yy = proj.meta.geometry.get_pixel_coords()
            
            min_x = min((min_x, min(xx)))
            min_y = min((min_y, min(yy)))
            max_x = max((max_x, max(xx)))
            max_y = max((max_y, max(yy)))
            
        # size of a block: 
        block_len = blocks[0].shape[1]     

        # Big slice:
        total_slice_shape = numpy.array([(max_y - min_y) / pixel[1], (max_x - min_x) / pixel[0]])             
        total_slice_shape = numpy.int32(numpy.ceil(total_slice_shape))         
        
        total_shape = numpy.array([total_slice_shape[0], block_len, total_slice_shape[1]])     
        total_centre = numpy.array([(max_y + min_y) / 2, (max_x + min_x) / 2])
                
        # Initialize a large projection array:
        big_block = numpy.zeros(total_shape, dtype = 'float32')
                
        #self.stich_tiles(self, total_slice, total_slice_size, total_centre, slice_key, offsets)
        weights = self._stich_slice(total_slice_shape, total_centre, blocks, pixel, -1)
        weights[weights < 0.1] = 0.1
          
        # Interpolate slice by slice:
        for ii in range(block_len): 
            big_block[:, ii, :] = self._stich_slice(total_slice_shape, total_centre, blocks, pixel, ii) / weights
            
        return big_block
        
    def _stich_slice(self, total_slice_shape, total_centre, blocks, pixel, slice_key):
        """
        Use interpolation to combine several tiles into one.
        """
        
        # Assuming all projections have equal number of angles and same pixel sizes
        total_slice_size = total_slice_shape * pixel        
        total_slice = numpy.zeros(total_slice_shape)

        # Block size:        
        sz = numpy.shape(blocks[0])
        
        for jj, proj in enumerate(self.projections):

            
            if slice_key == -1:
                # Initialize image of ones to compute weights:
                img = numpy.ones(sz[::2])
                
            else:    
                img = blocks[jj][:, slice_key, :]
            
            # Detector size in mm:
            det_size = proj.meta.geometry.det_size
                        
            # Offset from the left top corner:
            det_coords = numpy.flipud(proj.meta.geometry.det_trans[:-1])
            offset = ((det_coords - total_centre) + total_slice_size / 2 - det_size / 2) / pixel
            
            # Pad image to get the same size as the total_slice:        
            pad_x = total_slice.shape[1] - img.shape[1]
            pad_y = total_slice.shape[0] - img.shape[0]  
            img = numpy.pad(img, ((0, pad_y), (0, pad_x)), mode = 'constant')  
            img = interp.shift(img, offset, order = 1)
            
            total_slice += img
        
        return total_slice 
        
# TODO: random access
    
