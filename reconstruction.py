#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains tomographic reconstruction routines based on ASTRA.
"""
import astra
import data

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
        
        self.projections.extend(projections)

    def _backproject_block(self, proj_data, proj_geom, vol_data, vol_geom, algorithm, constraints = None):
        '''
        Backproject a single block of data
        '''
        
        cfg = astra.astra_dict(algorithm)
        cfg['option'] = {}

        # TODO:
        # Do we need to introduce ShortScan parameter?    
        
        if (constraints is not None):
          cfg['option']['MinConstraint'] = constraints[0]
          cfg['option']['MaxConstraint'] = constraints[1]
       
        try:
          rec_id = astra.data3d.link('-vol', vol_geom, vol_data)
          sinogram_id = astra.data3d.link('-sino', self.proj_geom, proj_data)
    
          cfg['ReconstructionDataId'] = rec_id
          cfg['ProjectionDataId'] = sinogram_id
             
          alg_id = astra.algorithm.create(cfg)
          
          # TODO: replace this with astra.experimental....
          astra.algorithm.run(alg_id)

        finally:
          astra.algorithm.delete(alg_id)
          astra.data3d.delete(rec_id)
          astra.data3d.delete(sinogram_id)
    
    def _forwardproject_block(self, proj_data, proj_geom, vol_data, vol_geom):
        '''
        Forwardproject a single block of data
        '''
        
        cfg = astra.astra_dict('FP3D_CUDA')
                
        try:          
                    
          rec_id = astra.data3d.link('-vol', vol_geom, vol_data)
          
          sinogram_id = astra.data3d.link('-sino', proj_geom, proj_data)
              
          cfg['VolumeDataId'] = rec_id
          cfg['ProjectionDataId'] = sinogram_id
    
          alg_id = astra.algorithm.create(cfg)
    
          astra.algorithm.run(alg_id, 1)

        finally:
          astra.algorithm.delete(alg_id)
          astra.data3d.delete(rec_id)
          astra.data3d.delete(sinogram_id)

    def backproject(self, algorithm = 'BP3D_CUDA', constraints = None):
        '''
        ASTRA backprojector. No filtering by default.
        '''
        
        # Volume geometry and projection geometry for ASTRA:
        vol_geom = self.volume.meta.get_vol_geom()
        
        # This will only work with RAM volume data! Otherwice, need to set data to 'total'.
        if isinstance(self.volume.data, data.data_blocks_swap):
            raise ValueError('Backprojection doesn`t support SSD data blocks for volumes!')
            
        # Pointer to the total volume:    
        vol_data = self.volume.data.total
        
        # Loop over different projection stacks:
        for proj in self.projections:        
            
            # Loop over blocks of data to save RAM:
            for block in proj.data:
                
                # ASTRA projection geometry for the current block:
                proj_geom = proj.meta.geometry.get_proj_geom(blocks = True)
                
                # Backprojection:
                self._backproject_block(block, proj_geom, vol_data, vol_geom, algorithm, constraints)    
        
    def forwardproject(self):
        '''
        ASTRA forwardprojector.
        '''
        
        # Volume geometry and projection geometry for ASTRA:
        vol_geom = self.volume.meta.get_vol_geom()
        
        # Loop over different projection stacks:
        for proj in self.projections:
            
            # This will only work with RAM volume data! Otherwise, need to use 'total' setter.
            if isinstance(proj.data, data.data_blocks_swap):
                raise ValueError('Forwardprojection doesn`t support SSD data blocks for projections!')
            
            proj_data = proj.data.total
        
            # ASTRA projection geometry:
            proj_geom = proj.meta.get_proj_geom()
            
            # Loop over blocks of data to save RAM:
            for block in self.volume.data:
                
                vol_geom = self.volume.meta.geometry.get_vol_geom(blocks = True)
                self._forwardproject_block(proj_data, proj_geom, block, vol_geom)    
        
    def FDK(self):
        '''
        The method of methods.
        '''
        # Switch to a different data storage if needed:
        self.volume.data.switch_to_ram(keep_data = True)
        
        #for proj in self.projections:
        #    
        #    if proj.data.sizeGB > 1:
        #        proj.data.switch_to_swap(keep_data = True)
        
        # Run the reconstruction:
        self.backproject(algorithm='FDK_CUDA')

        # Update history:    
        self.volume.meta.history.add_record('Reconstruction generated using FDK.', [])
    '''    
    def SIRT(self, iterations = 3):
        '''
        Simultaneous Iterative Reconstruction Technique... also known as James
        '''
        
        #weights = self.backproject(numpy.ones_like(data))
        
        #bwd_weights = 1.0 / sz[1]

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
        
        
    def CPLS(self):
        '''
        Chambolle-Pock Least Squares
        '''
        
    def EM(self):
        '''
        Expectation maximization
        '''
        
    def FISTA(self):
        '''
        '''
        
# TODO: random access
    '''
