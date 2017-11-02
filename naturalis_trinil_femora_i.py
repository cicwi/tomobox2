#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:50:38 2017

@author: kostenko
"""
#%%
import tomobox
import numpy
import reconstruction

#%%

#energy, spec = numpy.loadtxt('/export/scratch3/kostenko/Fast_Data/rijksmuseum/dummy/spectrum.txt')    

#index = [5,]    
index = numpy.arange(0, 8)    
path = '/export/scratch3/kostenko/Fast_Data/naturalis/oct2017_femora_trinil/femora_i/'
for ii in index:
    
    prj = tomobox.projections(block_sizeGB = 4, swap = False)
    #prj.data.switch_to_swap(swap_path = '/export/scratch1/kostenko/tomobox_swap', swap_name = 'swap')    
    prj.io.options['binning'] = 1
    prj.io.options['index_step'] = 1
    
    img_pixel = numpy.ones(3) * 0.045 * prj.io.options['binning']

    shape = numpy.array([1550,1450,1900], 'uint32') // prj.io.binning
    volume = tomobox.volume(shape = shape, img_pixel = img_pixel, block_sizeGB = 4)
    
    # Read data:
      
    
    
    prj.io.read_flexray(path + 'femora_i_tile_%u'%ii)

    # Process:
    prj.process.flat_field()
    prj.process.log()
    
    # Reconstruct:
    
    prj.meta.geometry.vol_trans[0] = 0
    prj.meta.geometry.vol_trans[2] = -10

    recon = reconstruction.reconstruct(prj, volume)
    recon.FDK()
    volume.display.projection(0)
        
    volume.io.write_data(path + ('femora_i_tile_%u' % ii) + '/FDK')
    
    recon = None
    volume.release()
    prj.release()
    
#%% STITCH
vol_pos = []
n = 8
for ii in range(n):
    vol_pos.append(ii*49)

vol_paths = []
for ii in range(n):
    vol_paths.append(path + 'femora_i_tile_%u/FDK/' % ii)

img_pix = 0.05
# Generate indexes:
indexes = []

for ii in range(n):
    offset = vol_pos[ii] - vol_pos[0]
    offset /= img_pix
    
    index = numpy.int32(numpy.round(numpy.arange(0, 1500) + offset))
    indexes.append(index)  
    
#%%
indexes    
#%% 

import data
import matplotlib.pyplot as plt
import os                  # file-name routines

for ii in range(4080, numpy.max(indexes)):
    
    print(ii)
    
    slices = []

    # Loop over volumes:
    for jj in range(n):
        index = indexes[jj]
        
        if ii in index:
            # Read volume slice
                        
            key = numpy.where(index == ii)[0][0]
            
            img = data.io._read_image(vol_paths[jj] + 'data_%05u.tiff'% key)
            
            slices.append(img)
            
    # Merge slices:
    if slices == []:
        break
    
    else:
        img = numpy.max(slices, 0)
        
        path1 = '/export/scratch3/kostenko/Fast_Data/naturalis/oct2017_femora_trinil/Femora_i_full_FDK/' 
        if not os.path.exists(path1):
            os.makedirs(path)
            
        img[img < 0] = 0
        ma = 8e-6
        img[img > ma] = ma
        img = img / ma * 255
        img = numpy.uint8(img)
        
        plt.imshow(img)
        plt.show()
        
        data.io._write_image(path1 + 'data_%05u.tiff'% ii, img)
        
#%% reload and bin:
import tomobox
#import numpy
#import reconstruction

volume = tomobox.volume(block_sizeGB = 4)

volume.io.options['binning'] = 2
volume.io.options['index_step'] = 2
#volume.io.read_slices('/export/scratch3/kostenko/Fast_Data/naturalis/oct2017_femora_trinil/Femora_i_full_FDK/') 
volume.io.read_slices('/run/media/kostenko/New Volume/rijks/bone/Femora_i_full_FDK/') 

volume.display.max_projection()
volume.display._cmap = 'gray'
volume.display.slice(430, dim = 1)

volume.display.projection(2)

#volume.io.write_data('/export/scratch3/kostenko/Fast_Data/naturalis/oct2017_femora_trinil/Femora_i_full_FDK_bin2/')

#%% Make a slice of a full resolution volume:
path = '/export/scratch3/kostenko/Fast_Data/naturalis/oct2017_femora_trinil/femora_i/'
ii = 5
volume.io.options['binning'] = 1
volume.io.options['index_step'] = 1
volume.io.read_slices(path + ('femora_i_tile_%u' % ii) + '/FDK')

volume.display.max_projection()
volume.display._cmap = 'gray'
volume.display._dynamic_range = [0, 4e-6]
volume.display.slice(430*2, dim = 1)

volume.display.projection(2)
#volume.release()     