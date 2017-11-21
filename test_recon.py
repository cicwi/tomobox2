# -*- coding: utf-8 -*-
"""
Test of the reconstructor.    
"""

#%% Test IO

import tomobox
import reconstruction
import simulate

#%% Create aphantom:
#phantom = simulate.phantom.shepp3d([10, 256, 256])
phantom = simulate.phantom.checkers([10, 256, 256])
vol = tomobox.volume(array = phantom, img_pixel = [0.1,0.1,0.1], block_sizeGB = 3)
vol.display.slice()

#%% Forward project:  
proj = tomobox.projections(block_sizeGB = 3)    
proj.data.zeros([20, 100, 256])
proj.meta.geometry.init(100, 100, [0.1, 0.1], theta_n = 100)
    
recon = reconstruction.reconstruct(proj, vol)
recon.forwardproject()

proj.display.slice()

#%% Test Back-projector:
vol.data.zeros()
#recon._projections_to_swap()

recon.backproject(multiplier = 1)
vol.display.slice()



















#%% Test Forward projector:

proj.data.total = proj.data.total * 0

#proj.data.make_contiguous()

recon._vol_to_swap()
recon.forwardproject(multiplier = -1)

proj.display.slice(dim = 1)

#%% TestFDK:

pix = proj.meta.geometry.img_pixel[1]
vol = tomobox.vol([100,100,100], pix)

recon = reconstruction.reconstruct(proj, vol)
recon.FDK()

vol.display.slice(dim = 0)
vol.display.slice(dim = 1)
    
#%% 
import reconstruction
plt.close('all')

vol = tomobox.volume([100,100,100], 0.3)
vol.data.zeros()
recon = reconstruction.reconstruct(proj, vol)

vol.data.block_key = -1
proj.data.block_key = -1
recon.SIRT(25)

#vol.display.slice(dim = 0)
#vol.display.slice(dim = 1)

#%%



#%% SIRT OPEN:
vol = tomobox.volume([100,100,100], 0.3)
w = recon.volume.data.shape.max() 
recon = reconstruction.reconstruct(proj, vol)
plt.close('all')
proj.display.slice(dim = 1)

# One iteration: 
recon._volume_to_swap()
recon.forwardproject(multiplier = -1/10)
proj.display.slice(dim = 1) 
     
recon._projections_to_swap()
recon.backproject(multiplier = 1/w)


recon.volume.display.slice(dim = 0)





# Switch data to swap if needed:
proj.display.slice(dim = 1)  
recon.volume.display.slice(dim = 0)





#%% Forward projection:    
w = self.volume.data.shape.max() 

for ii in range(iterations):
    
    # Preview:
    self.volume.display.slice(dim = 0, fig_num = 11)
    self.projections[0].display.slice(dim = 1, fig_num = 13)

    #print('****', self.volume.data.total.sum())            
    # Switch data to swap if needed:
    self._volume_to_swap()
    
    #print('*****', self.volume.data.total.sum())            
    print('vol before forwardproject', self.volume.data.total.sum()) 
    self.forwardproject(multiplier = -1 / 10)
    print('vol after forwardproject', self.volume.data.total.sum()) 

    print('vol before _projections_to_swap', self.volume.data.total.sum()) 
    self._projections_to_swap()
    print('vol after _projections_to_swap', self.volume.data.total.sum()) 
    
    self.backproject(multiplier = 1 / w)
