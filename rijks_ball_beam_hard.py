# -*- coding: utf-8 -*-
"""
Test of the reconstructor.   

No swap. Use 
"""

#%% Test IO
%pylab
import tomobox

# On weld scratch1 seems to be the ssd drive
prj = tomobox.projections(block_sizeGB = 1, swap = False)
    
#%% Read and process:
   
prj.io.binning = 4
prj.io.read_flexray('/export/scratch3/kostenko/Fast_Data/rijksmuseum/AK-NM-7019/t%u'%(ii))

# Process:
prj.process.flat_field()
prj.process.log()
prj.display.slice(dim = 1)
    
energy, spec = numpy.loadtxt('/export/scratch3/kostenko/Fast_Data/rijksmuseum/dummy/spectrum.txt')
prj.process.equivalent_thickness(energy, spec, 'Bone, Compact (ICRU)', 1.6)

prj.display.slice(dim = 1)
