# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 09:54:38 2017

@author: kostenko
"""
import numpy

def report_geometry(geometry):
        '''
        Print a report of the geometry.
        '''
        print('   *** GEOMETRY REPORT ***')
        print('________________________________')
        print('   Base geometry:')
        print('________________________________')
        print('Source to Object Distance: %2.2f mm' % geometry.src2obj)
        print('Detector to Object Distance: %2.2f mm' % geometry.det2obj)
        print('Detector Pixel: %2.2f x %2.2f mm' % (geometry.det_pixel[1], geometry.det_pixel[0]))
        print('Magnification: %2.2f' % geometry.magnification)
        print('Detector size: %u x %u pixels' % (geometry._parent.data.shape[2],geometry._parent.data.shape[0]))
        print('Detector physical size: %2.2f x %2.2f mm' % (geometry.det_size[1], geometry.det_size[0]))
        print('Rotation range: [%2.2f , %2.2f ]' % (geometry.theta_range[0], geometry.theta_range[1]))
        print('Rotation step: %2.2f' % geometry.theta_step)
        
        # in case it's a flex_geometry:
        try:
           print('________________________________')
           print('   Flexray geometry:')
           print('________________________________')
           print('Detector translation [hor, vert, mag]: %2.2f, %2.2f, %2.2f mm' % (geometry.det_trans[0], geometry.det_trans[1], geometry.det_trans[2]))
           print('Detector rotation (in detector plane): %2.2f rad' % geometry.det_rot)
           
           print('Source translation [hor, vert, mag]: %2.2f, %2.2f, %2.2f mm' % (geometry.src_trans[0], geometry.src_trans[1], geometry.src_trans[2]))
           
           print('Rotation axis translation [hor, mag]: %2.2f, %2.2f mm' % (geometry.axs_trans[0], geometry.axs_trans[1]))
           
           print('Volume translation [hor, vert, mag]: %2.2f, %2.2f, %2.2f mm' % (geometry.vol_trans[0],geometry.vol_trans[1],geometry.vol_trans[2]))
           print('Volume  rotation [Euler]: %2.2f, %2.2f, %2.2f rad' % (geometry.vol_rot[0], geometry.vol_rot[1], geometry.vol_rot[2]))
           
           if numpy.max(numpy.abs(geometry.theta_offset)) > 0:
               print('Theta deviation: ', geometry.theta_offset)
               
           print('')    
           
        except AttributeError:
           print('End of the geometry report (flex_geometry unavailable)')
           

    