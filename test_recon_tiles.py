# -*- coding: utf-8 -*-
"""
Test of the reconstructor.    
"""

#%% Test IO

import tomobox

a = tomobox.projections(block_sizeGB = 1, swap = False)
a.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/tiling/al_dummy_vertical_tile_0/')

a.process.flat_field()
a.process.bin_projections()
a.process.log()
a.display.slice(dim = 1)

a.meta.geometry.report()    





b = tomobox.projections(block_sizeGB = 1, swap = False)
b.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/tiling/al_dummy_vertical_tile_1/')

b.process.flat_field()
b.process.bin_projections()
b.process.log()
b.display.slice(dim = 1)

b.meta.geometry.report()    

b.data.switch_to_swap(keep_data=True, swap_path = '/export/scratch3/kostenko/Fast_Data/swap', swap_name = 'swap_0')
b.data.switch_to_swap(keep_data=True, swap_path = '/export/scratch3/kostenko/Fast_Data/swap', swap_name = 'swap_1')

#%% QC:
proj0.meta.geometry.report()    
proj1.meta.geometry.report()    
#%% Test 

import reconstruction

pix = proj0.meta.geometry.img_pixel[1]
volume = tomobox.volume([400,400,400], pix)

recon = reconstruction.reconstruct([proj0, proj1], volume)


