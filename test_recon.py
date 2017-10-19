# -*- coding: utf-8 -*-
"""
Test of the reconstructor.    
"""

#%% Test IO

import tomobox

proj = tomobox.projections()
proj.switch_to_swap(block_sizeGB = 0.5)

#proj.io.read_flexray('/export/scratch2/kostenko/Slow_Data/Rijksmuseum/misc/fingerprint/')
proj.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/tiling/al_dummy_vertical_tile_0')

#%% Test process

proj.process.flat_field()
proj.process.log()

proj.display.max_projection()

#%% Test 

import reconstruction

pix = proj.meta.geometry.img_pixel[1]
volume = tomobox.volume([1000,1000,1000], pix)

recon = reconstruction.reconstruct(proj, volume)

recon.FDK()

volume.display.slice(dim=0)


#%%
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(vol_data[:, 500, :])
#plt.imshow(vol_data[200, :, :])
plt.colorbar()
