# -*- coding: utf-8 -*-
"""
Lets test the projections class here.

Related classes are:
    
    data_blocks
    
    io
    processing
    display
    analysis
    
"""


#%% Test data_blocks
import data
import numpy

db = data.data_array(shape = [1000, 1000, 1000], block_sizeGB = 0.5, dim = 0)

print('GB', db.sizeGB)

print(db.block_number)
print(db.slice_shape)
print(db.block_shape)

print(db.empty_slice(1).shape)
print(db.empty_block(1).shape)
#print(db.block_xyz())
print(db.get_slice(1).shape)


blo0 = db.empty_block()
for blo in db:
    blo0 += numpy.log(blo+1)

del blo0

#%% Test IO
import tomobox

proj = tomobox.projections()

# Work:
#proj.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/tiling/al_dummy_vertical_tile_0')

# Home:
proj.io.read_flexray('D:/Data/al_dummy_vertical_tile_1')
    
proj.display.slice()
proj.display.projection()

#%% Test process
proj.process.flat_field()
proj.process.log()
proj.process.residual_rings()
proj.process.salt_pepper()
proj.process.add_noise(1, mode = 'normal')

#%% Test swap:
import tomobox

proj = tomobox.projections(block_sizeGB = 0.5, swap = True)
proj.data.switch_to_swap(swap_path = 'D:/Data/swap', swap_name = 'swap')

# Work:
#proj.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/tiling/al_dummy_vertical_tile_0')

# Home:
proj.io.read_flexray('D:/Data/al_dummy_vertical_tile_1')
    
proj.display.slice()
proj.display.projection()    

#%% Random access:
print(proj.data.block_index)
proj.data.set_indexer('random')
print(proj.data.block_index)
proj.data.set_indexer('equidistant')
print(proj.data.block_index)
#%% Reconstruct!
