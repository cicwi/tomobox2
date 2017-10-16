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

db = data.data_blocks(numpy.zeros([2000,500,1000]), block_sizeGB = 1)
db.dim = 0

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

proj.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/tiling/al_dummy_vertical_tile_0')

proj.display.slice()
proj.display.projection()

#%% Test process
proj.process.flat_field()
proj.process.log()
proj.process.residual_rings()
proj.process.salt_pepper()
proj.process.add_noise(1, mode = 'normal')

#%% Test SSD

import tomobox

# Create data blocks and populate:
proj = tomobox.projections()
proj.switch_to_swap()

# Create data blocks and populate block by block:
proj.data[0] = proj.data.empty_block() + 1
proj.data[1] = proj.data.empty_block() + 2
proj.data[2] = proj.data.empty_block() + 3
proj.data[3] = proj.data.empty_block() + 4

#%% 
import tomobox

# Create data blocks and populate:
proj = tomobox.projections()
proj.switch_to_swap()

proj.data.total = numpy.zeros([1000,1000,1000])

for ii, blk in enumerate(proj.data):
    blk = blk + ii
    proj.data = blk
    

#%% Test process on SSD
import tomobox

# Create data blocks and populate:
proj = tomobox.projections()
proj.switch_to_swap()
proj.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/tiling/al_dummy_vertical_tile_0')

#%%
proj.process.flat_field()
proj.process.log()
proj.process.residual_rings()
#%% Reconstruct!
