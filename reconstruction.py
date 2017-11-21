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
import matplotlib.pyplot as plt

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
    
        self.options = {'swap': False, 'multiblock': False, 'constraints': None, 'poisson_weight': False, 'ramp': 0, 'backproject_weight': False, 'preview': False, 'L2_update': True}

        self.swap_path = '/export/scratch3/kostenko/Fast_Data/swap'
    
        # Remember the projection data that we will use to reconstruct:
        self.add_projections(projections)
        
        # This is your volume:
        self.volume = volume           
        
        # L2 norm of the residual:
        self.l2 = 0
        
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
                            
    def _backproject_block(self, proj_data, proj_geom, vol_data, vol_geom, algorithm):
        '''
        Backproject a single block of data
        '''
        
        # TODO:
        # Do we need to introduce ShortScan parameter?    
        # At the moment we will ignore MinConstraint / MaxConstraint    
                
        try:
            sin_id = astra.data3d.link('-sino', proj_geom, numpy.ascontiguousarray(proj_data))
            
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

    def backproject(self, volume = None, algorithm = 'BP3D_CUDA', multiplier = 1):
        '''
        ASTRA backprojector. No filtering by default.
        It will loop over all registered projection datasets, 
        collect one data block from each at a time and project it.
        '''
        if volume == None:
            volume = self.volume
            
        # Loop over different projection stacks:
        for proj in self.projections:  
   
            misc.progress_bar(0)        
            
            # Loop over blocks of data to save RAM:
            for ii, proj_data in enumerate(proj.data):
                
                slice_shape  = proj.data.slice_shape
                
                # ASTRA projection geometry for the current block:
                proj_geom = proj.meta.geometry.get_proj_geom(slice_shape, blocks = True)
                
                #print('vol_data', vol_data.sum())
                
                if multiplier != 1:
                    self._backproject(proj_data * multiplier, proj_geom, volume, algorithm = algorithm)
                else:
                    self._backproject(proj_data, proj_geom, volume, algorithm = algorithm)
                    
            misc.progress_bar((ii+1) / proj.data.block_number)        

    def _backproject(self, proj_data, proj_geom, volume, algorithm = 'BP3D_CUDA'):
        """
        Back-project to all volume blocks or the total volume.
        """
        # Single block volume:
            
        if self.options['multiblock'] == False:   
                    
            vol_data = volume.data.total
            vol_geom = volume.meta.geometry.get_vol_geom()  
            
            self._backproject_block(proj_data, proj_geom, vol_data, vol_geom, algorithm)     
            
            # Constraints:
            constr = self.options['constraints']
            if constr is not None:
                numpy.clip(vol_data, a_min = constr[0], a_max = constr[1], out = vol_data) 
            
            
            volume.data.total = vol_data 
        
        # Multi-block volume:    
        else:    
            
            print('Reconstructing multiblock')
        
            # Back-Project into all volume blocks:
            for jj, vol_data in enumerate(volume.data):
                
                # Back-project the residual:
                vol_geom = volume.meta.geometry.get_vol_geom(blocks = True)  
                self._backproject_block(proj_data, proj_geom, vol_data, vol_geom, algorithm)
                
                # Constraints:
                constr = self.options['constraints']
                if constr is not None:
                    numpy.clip(vol_data, a_min = constr[0], a_max = constr[1], out = vol_data) 
                    
                volume.data[jj] = vol_data
                                                    
    def stich_tiles(self, projection_data):
        """
        Stitch several tiles together and produce a new projection data. Output into projection_data.
        """ 

        #print('Stitching tiles...')               
        
        # Total data shape:
        slice_shape, centre = self._stitch_shape()    
        data_length = self.projections[0].data.length
        data_shape = [slice_shape[0], data_length, slice_shape[1]]
        
        projection_data.data.init_total(data_shape)
        
        # Assuming all data has the same amount of blocks and projections    
        bn = self.projections[0].data.block_number     

        # Make sure that the block length of the stiched data is the same as the input:
        projection_data.data.change_block_length(self.projections[0].data.block_shape[1])
        
        print('New block length is ', self.projections[0].data.block_shape[1])
        print('New block sizeGB is ', projection_data.data.sizeGB)

        for ii in range(bn):   
            # Blocks from different datasets:
            blocks = []

            # geometries
            geometries = []

            # Loop over different projection stacks:
            for proj in self.projections:  
                
                blocks.append(proj.data[ii])
                geometries.append(proj.meta.geometry)
   
            pixel = self.projections[0].meta.geometry.det_pixel

            # Produce one big block:
            big_block = self._stitch_block(blocks, pixel)                
            projection_data.data[ii] = big_block

            misc.progress_bar((ii+1) / bn)

        # Create a projection geometry:            
        slice_shape = projection_data.data.slice_shape

        print('New slice shape is:', slice_shape)

        big_geom = flex_geometry.mean(geometries)
        
        # This is important, new geometry should know it's parent!
        big_geom._parent = projection_data
        
        projection_data.meta.geometry = big_geom   
            
    def backproject_tiles(self, algorithm = 'BP3D_CUDA', multiplier = 1):
        '''
        ASTRA backprojector. Will stitch tiles before backprojection.
        '''
        
        import matplotlib.pyplot as plt
        
        # This will only work with RAM volume data! Otherwice, need to set data to 'total'.
        #if self.volume.data.is_swap():
        #    raise ValueError('Backprojection doesn`t support swap data array for volumes!')
                    
        # Assuming all data has the same amount of blocks and projections    
        bn = self.projections[0].data.block_number  

        misc.progress_bar(0)                    

        for ii in range(bn):
            
            # Blocks from different datasets:
            blocks = []

            # proj_geometries
            geometries = []

            # Loop over different projection stacks:
            for proj in self.projections:  
                
                blocks.append(proj.data[ii])
                geometries.append(proj.meta.geometry)
   
            pixel = self.projections[0].meta.geometry.det_pixel

            # Produce one big block:
            # print('Stitching a block...')    
            
            big_block = self._stitch_block(blocks, pixel)    

            #print('Total block shape is', big_block.shape)
            
            slice_shape = big_block.shape[::2]

            big_geom = flex_geometry.mean(geometries).get_proj_geom(slice_shape, blocks= True)
            
            plt.imshow(big_block[:,0,:])
            plt.show()
            
            print('Backprojection...') 
            
            if multiplier != 1:
                self._backproject(multiplier * big_block, big_geom, self.volume, algorithm = algorithm)
            else:
                self._backproject(big_block, big_geom, self.volume, algorithm = algorithm)
            
            misc.progress_bar((ii+1) / bn)    
                
    def _forwardproject_block(self, proj_data, proj_geom, vol_data, vol_geom):
        '''
        Forwardproject a single block of data
        '''
        try:  
          sin_id = astra.data3d.link('-sino', proj_geom, proj_data)  
          vol_id = astra.data3d.link('-vol', vol_geom, vol_data)         
          
          projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
          
          #print('**proj_data', proj_data.sum())
          asex.accumulate_FP(projector_id, vol_id, sin_id) 
          #print('**proj_data', proj_data.sum())            
        except:
          print("ASTRA error:", sys.exc_info())  
          
        finally:
          astra.data3d.delete(sin_id)
          astra.data3d.delete(vol_id)
          astra.projector.delete(projector_id)
          
        return proj_data   

    def compare_block(self):
        '''
        Forward project one block and compare with the sinogram.
        '''                      
        proj = self.projections[0]            
        proj_data = proj.data[0]

        synth_proj_data = proj.data.empty_block()
                       
        slice_shape = proj.data.slice_shape
        
        # ASTRA projection geometry:
        proj_geom = proj.meta.geometry.get_proj_geom(slice_shape, blocks = True)
        
        self._forwardproject(synth_proj_data, proj_geom, self.volume, 1)
        
        print('Computing the factor!')
        
        factor = proj_data / synth_proj_data
        factor = factor[synth_proj_data > synth_proj_data.max()/10]
        factor = factor[numpy.isfinite(factor)]
        
        return factor.mean()
                
    def forwardproject(self, volume = None, multiplier = 1):
        '''
        ASTRA forwardprojector. 
        '''
        if volume == None:
            volume = self.volume
        
        # Loop over different projection stacks:
        for proj in self.projections:
            
            # Loop over blocks:
            for jj, proj_data in enumerate(proj.data):
                
                # Make sure that our projection data pool is not updated by the forward projection:
                #proj.data._read_only = True
                if proj_data.size == 0:
                  raise ValueError('Projection data array is empty. Cannot forwardproject into an empty array.')
            
                slice_shape  = proj.data.slice_shape
                
                # ASTRA projection geometry:
                proj_geom = proj.meta.geometry.get_proj_geom(slice_shape, blocks = True)
                
                self._forwardproject(proj_data, proj_geom, volume, multiplier)
                
                proj.data[jj] = proj_data
                
        
    def _forwardproject(self, proj_data, proj_geom, volume, multiplier = 1):
        """
        Forward project to all volume blocks or the total volume.
        """
        # Single block volume:
        if self.options['multiblock'] == False: 
            
            vol_data = volume.data.total
            vol_geom = volume.meta.geometry.get_vol_geom()  
            
            self._forwardproject_block(proj_data, proj_geom, vol_data * multiplier, vol_geom)     
        
        # Multi-block volume:    
        else:    
        
            # Project all volume blocks:
            for vol_data in volume.data:    
                
                # Volume geometry and projection geometry for ASTRA:
                vol_geom = volume.meta.geometry.get_vol_geom(blocks = True)  
                
                # Forward project and subtract from the buffer of proj_data               
                self._forwardproject_block(proj_data, proj_geom, vol_data * multiplier, vol_geom)     
                
    def _volume_to_swap(self, read_only = False):
        '''
        Put volume to swap and projections to RAM. Do it before forwardprojection
        '''        
        if self.options['swap']:
            self.volume.data.switch_to_swap(keep_data = True, swap_path = self.swap_path, swap_name = 'vol_swap')
            
        self.volume.data._read_only = read_only
            
    def _projections_to_swap(self, read_only = False):
        '''
        Put projections to swap and volume to RAM. Do it before backprojection
        '''
        if self.options['swap']:
            for proj in self.projections:
                proj.data.switch_to_swap(keep_data = True, swap_path = self.swap_path, swap_name = 'proj_swap')
                
        for proj in self.projections:
            proj.data._read_only = read_only    
            
    def _compute_forward_weight(self):
        '''
        Compute weights applied to the forward projection. FP(BP(ONES))
        '''          
                       
    def FDK(self, norm_forward = False):
        '''
        The method of methods.
        '''
        
        #print('Switching volume data to RAM')
        
        # Switch to a different data storage if needed:
        #self.volume.data.switch_to_ram(keep_data = False)
        
        # Make sure you start form a blank volume
        self.volume.data.zeros()
        
        print('Computing backprojection')
        
        # Run the reconstruction:
        if len(self.projections) > 1:
            self.backproject_tiles(algorithm='FDK_CUDA')
            
        else:
            self.backproject(algorithm='FDK_CUDA')

        if norm_forward:
            # Compute normalization: 
            fact = self.compare_block()
            self.volume.data.multiply(fact)
            
        # Reverse ingeneering shows that FDK should be scaled like this:
        #img_pixel = self.volume.meta.geometry.img_pixel
        #thetaN = 1e-3 * 0.84#self.projections[0].meta.geometry.theta_n * 1e-4
        #M = 2#self.projections[0].meta.geometry.magnification
        #N = self.volume.data.shape[1]         

        #self.volume.data.divide(N * img_pixel[0] * img_pixel[1]**2 * img_pixel[2] * thetaN / M)
    
        # Update history:    
        self.volume.meta.history.add_record('Reconstruction generated using FDK.', [])
                
    def SIRT(self, iterations = 5):
        '''
        Simultaneous Iterative Reconstruction Technique using subsets... also known as James
        '''        

        # Weight to be on the safe side:
        #prj_weight = 1 / numpy.sqrt(self.volume.data.shape.max()) * self.projections[0].data.block_number
        
        astra_norm = self.projections[0].data.length * self.volume.meta.geometry.img_pixel[0]**4 * self.volume.data.shape.max()
        
        prj_weight = 2 / astra_norm * (self.projections[0].data.block_number)

        # Initialize a reconstruction volume:
        #guess = self.volume.copy(swap = False)   
        # Make sure that our guess volume is a single block RAM based:
        #guess.data.switch_to_ram()
        #guess.data.change_block_size(9999)
                        
        # Initialize L2:
        l2 = []

        volume = self.volume
        
        print('Doing SIRT`y things...')
        
        misc.progress_bar(0)        
        for ii in range(iterations):
        
            # Here loops of blocks and projections stack can be organized diffrently
            # I will implement them in the easyest way now.
            _l2 = 0
            # Loop over different projection stacks:
            for proj in self.projections:
                
                slice_shape = proj.data.slice_shape
                            
                # Loop over blocks:
                for proj_data in proj.data:
                    
                    # Make sure that our projection data pool is not updated by the forward projection
                    # update the buffer only to keep residual in it
                    proj.data._read_only = True
                    
                    # Geometry of the block:
                    proj_geom = proj.meta.geometry.get_proj_geom(slice_shape, blocks = True)
                    
                    # Forward project:    
                    self._forwardproject(proj_data, proj_geom, volume, -1)    
                    
                    # Take into account Poisson:
                    if self.options['poisson_weight']:
                        # Some formula representing the effect of photon starvation...
                        proj_data *= numpy.sqrt(numpy.exp(-proj_data)) * prj_weight 

                    else:
                        # Apply weights to forward projection residual:
                        proj_data *= prj_weight
                        
                    # L2 norm:
                    if self.options['L2_update']:
                        #print('L2',numpy.sqrt((proj_data ** 2).mean()))
                        _l2 += numpy.sqrt((proj_data ** 2).mean()) / proj.data.block_number 
                    
                    self._backproject(proj_data, proj_geom, volume)
                    
                proj.data._read_only = False
                
            l2.append(_l2)
            
            # Preview
            if self.options['preview']:
                self.volume.display.slice(dim = 0)
                
            misc.progress_bar((ii+1) / iterations)
        
        plt.figure(15)
        plt.plot(l2)
        plt.title('Residual L2')
        
        self.volume.meta.history.add_record('Reconstruction generated using SIRT.', iterations)
                
    def CPLS(self):
        '''
        Chambolle-Pock Least Squares
        '''
        pass
        
    def EM(self, iterations = 5):
        '''
        Expectation maximization
        '''
        # Weight to be on the safe side:
        #astra_norm = self.projections[0].data.length * self.volume.meta.geometry.img_pixel[0]**4 * self.volume.data.shape.max()
        #prj_weight = 2 / astra_norm * (self.projections[0].data.block_number)
        prj_weight = 1
                        
        # Initialize a reconstruction volume:
                        
        # Initialize L2:
        #norm = []

        volume = self.volume
        
        # Make sure volume is above null...
        numpy.clip(volume.data.total, a_min = 0, a_max = 99, out = volume.data.total)
        #volume.data.total *= numpy.int32(volume.data.total > 0)
        
        print('Em Emm Emmmm...')
        
        misc.progress_bar(0)        
        for ii in range(iterations):
        
            # Here loops of blocks and projections stack can be organized diffrently
            # I will implement them in the easyest way now.
            #_norm = 0
            # Loop over different projection stacks:
            for proj in self.projections:
                
                slice_shape = proj.data.slice_shape
                            
                # Loop over blocks:
                for proj_data in proj.data:
                    
                    # Make sure that our projection data pool is not updated by the forward projection
                    # update the buffer only to keep residual in it
                    proj.data._read_only = True
                    
                    # Geometry of the block:
                    proj_geom = proj.meta.geometry.get_proj_geom(slice_shape, blocks = True)
                    
                    # Take into account Poisson:
                    #if self.options['poisson_weight']:
                    #    prj_weight = prj_weight * numpy.sqrt(numpy.exp(-proj_data))
                    
                    # Empty block:    
                    block = proj.data.empty_block()
                    
                    # Forward project:    
                    self._forwardproject(block, proj_geom, volume)    
                    
                    block[block < 0.0001] = numpy.inf

                    # Norm:
                    #if self.options['L2_update']:
                    #    _norm += numpy.sqrt(((proj_data - block) ** 2).mean()) / proj.data.block_number 
                    
                    # Divide and regularize:
                    proj_data /= block
                    
                    #numpy.clip(proj_data, a_min = 0.1, a_max = 10, out = proj_data) 
                    
                    # Apply weights to forward projection residual:
                    proj_data *= prj_weight
                    
                    # Backproject and multiply:
                    # Can't use standard _backproject method because of the multiplication:
                    if self.options['multiblock']:
                        for jj, vol_data in enumerate(volume.data):
                            
                            # Back-project the residual:
                            vol_geom = volume.meta.geometry.get_vol_geom(blocks = True) 
                            
                            vol_block = volume.data.empty_block()
                            self._backproject_block(proj_data, proj_geom, vol_block, vol_geom, 'BP3D_CUDA')
                            
                            vol_data *= vol_block
                            
                            # Constraints:
                            constr = self.options['constraints']
                            if constr is not None:
                                numpy.clip(vol_data, a_min = constr[0], a_max = constr[1], out = vol_data) 
                            
                            volume.data[jj] = vol_data
                    else:
                        
                        # Back-project the residual:
                        vol_geom = volume.meta.geometry.get_vol_geom() 
                        vol_data = numpy.zeros_like(volume.data.total)
                        
                        self._backproject_block(proj_data, proj_geom, vol_data, vol_geom, 'BP3D_CUDA')
                        
                        # Constraints:
                        constr = self.options['constraints']
                        if constr is not None:
                            numpy.clip(vol_data, a_min = constr[0], a_max = constr[1], out = vol_data) 
                        
                        volume.data.total *= vol_data

                proj.data._read_only = False
                
            #norm.append(_norm)
            
            # Preview
            if self.options['preview']:
                self.volume.display.slice(dim = 0)
                
            misc.progress_bar((ii+1) / iterations)
        
        #plt.figure(15)
        #plt.plot(norm)
        #plt.title('Residual L2')
        
    def FISTA(self):
        '''
        
        '''
        pass
        
    def _stitch_shape(self):
        """
        Compute the size of the stiched dataset.
        """
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
        
        pixel = self.projections[0].meta.geometry.det_pixel    
            
        # Big slice:
        slice_shape = numpy.array([(max_y - min_y) / pixel[1], (max_x - min_x) / pixel[0]])             
        slice_shape = numpy.int32(numpy.ceil(slice_shape))  
        
        total_centre = numpy.array([(max_y + min_y) / 2, (max_x + min_x) / 2])
        
        return slice_shape, total_centre
    
    def _stitch_block(self, blocks, pixel):
        '''
        Stich several projection tiles into one projection.
        
        '''        
        # size of a block: 
        block_len = blocks[0].shape[1] 
        total_slice_shape, total_centre = self._stitch_shape()

        total_shape = numpy.array([total_slice_shape[0], block_len, total_slice_shape[1]])     
                
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
    
