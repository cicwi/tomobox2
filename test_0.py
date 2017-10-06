"""
Testing the data routines.
"""
#%%

x = data_blocks(numpy.ones([1000,1000,1000]), 1)

x.block_number

#%%
for data in x:
    print(data.shape)
    
#%%

x = data_blocks(numpy.ones([3,3,3]), 1)

x[0]    