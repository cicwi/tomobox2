# -*- coding: utf-8 -*-
"""
Test of the reconstructor.    
"""

#%% Test IO

import tomobox

proj = tomobox.projections(block_sizeGB = 1, swap = False)
#proj.data.switch_to_swap(swap_path = 'D:/Data/swap', swap_name = 'swap')

proj.io.read_flexray('/export/scratch2/kostenko/Slow_Data/Rijksmuseum/misc/fingerprint/')
#proj.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/tiling/al_dummy_vertical_tile_0')
#proj.io.read_flexray('D:/Data/al_dummy_vertical_tile_1')

#%%
proj.meta.geometry.report()

#%% Test process

proj.process.flat_field()
proj.process.bin_projections()
proj.process.log()
proj.display.slice(dim = 1)

#%% Test 

import reconstruction

pix = proj.meta.geometry.img_pixel[1]
volume = tomobox.volume([400,400,400], pix)

recon = reconstruction.reconstruct(proj, volume)
recon.FDK()

volume.display.slice(dim = 0)
volume.display.slice(dim = 1)

#%% 
volume.data.zeros()
recon = reconstruction.reconstruct(proj, volume)

recon.SIRT(5)

volume.display.slice(dim = 0)
volume.display.slice(dim = 1)

#%%
proj.data.total = proj.data.total * 0
recon.projections[0].data.switch_to_ram(keep_data=True)

recon.forwardproject(multiplier = -1)

proj.display.slice(dim = 1)
proj.analyse.sum()

