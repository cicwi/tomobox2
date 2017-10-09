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

db = data_blocks()
db.total = numpy.zeros([1000,1000,1000], dtype = 'float32')
db.block_size = 1

print(db.sizeGB)
print(db.block_number)
print(db.empty_slice(1))
print(db.block_xyz())
print(db.get_slice(1))

for blo in db:
    pass


get slice with sampling
#%% Test IO:
    
projection.    
