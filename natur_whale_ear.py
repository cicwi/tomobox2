# -*- coding: utf-8 -*-
"""
Test of the reconstructor.   

No swap. Use 
"""

#%% Test IO
%pylab
import tomobox
import reconstruction

# On weld scratch1 seems to be the ssd drive
prj = tomobox.projections(block_sizeGB = 4, swap = False)
    
#%% Read and process:
bine = 1
prj.io.options['binning'] = bine
prj.io.options['index_step'] = 1
prj.io.read_flexray('/export/scratch3/kostenko/Fast_Data/naturalis/oct2017_femora_trinil/whale_ear_1/')

# Process:
prj.process.flat_field()
prj.process.log()
prj.display.slice(dim = 1)
    
#energy, spec = numpy.loadtxt('/export/scratch3/kostenko/Fast_Data/rijksmuseum/dummy/spectrum.txt')
#prj.process.equivalent_thickness(energy, spec, 'Bone, Compact (ICRU)', 1.6)

prj.meta.geometry.report()

#%%
shape = numpy.array([1500,1900,1900]) // bine 
img_pixel = numpy.array([0.03, 0.03, 0.03]) * bine

volume = tomobox.volume(shape = shape, img_pixel = img_pixel)
recon = reconstruction.reconstruct(prj, volume)

recon.FDK()

volume.data.cast_uint8()
volume.io.write_data('/export/scratch3/kostenko/Fast_Data/naturalis/oct2017_femora_trinil/whale_ear_1/'+ 'FDK')

volume.display.slice()

#%% Release:

volume.release()
prj.release()
