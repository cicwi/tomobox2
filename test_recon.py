# -*- coding: utf-8 -*-
"""
Test of the reconstructor.    
"""

#%% Test IO

import tomobox

proj = tomobox.projections(block_sizeGB = 0.5, swap = True)
proj.data.switch_to_swap(swap_path = 'D:/Data/swap', swap_name = 'swap')

#proj.io.read_flexray('/export/scratch2/kostenko/Slow_Data/Rijksmuseum/misc/fingerprint/')
#proj.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/tiling/al_dummy_vertical_tile_0')
proj.io.read_flexray('D:/Data/al_dummy_vertical_tile_1')

#%% Test process

proj.process.flat_field()
# last slice?

proj.process.bin_projections()
# we have an unfinalized slice?

proj.process.log()

proj.display.max_projection()

#%% Test 

import reconstruction

pix = proj.meta.geometry.img_pixel[1]
volume = tomobox.volume([800,800,800], pix)

recon = reconstruction.reconstruct(proj, volume)
recon.swap_path = 'D:/Data/swap'

recon.backproject()
#recon.FDK()

volume.display.slice(dim=0)
