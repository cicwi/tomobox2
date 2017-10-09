# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 09:54:38 2017

@author: kostenko
"""

def report(geometry):
        '''
        Print a report of the geometry.
        '''
        print('   *** GEOMETRY REPORT ***')
        print('_______________')
        print(' All values, except for the detector size and pixel size, are in pixel sizes.')
        print('_______________')
        print('   Base geometry:')
        print('_______________')
        print('Source to Object Distance: %2.2f mm' % geometry.src2obj)
        print('Detector to Object Distance: %2.2f mm' % geometry.det2obj)
        print('Detector Pixel: %2.2f x %2.2f mm' % (geometry.det_pixel[1], geometry.det_pixel[0]))
        print('Magnification: %2.2f' % geometry.magnification)
        print('Detector size: %u x %u pixels' % (geometry._parent.data.shape[2],geometry._parent.data.shape[0]))
        print('Detector physical size: %2.2f x %2.2f mm' % (geometry.det_size[1], geometry.det_size[0]))
        print('Rotation range: [%2.2f , %2.2f ]' % (geometry.theta_range[0], geometry.theta_range[1]))
        print('Rotation step: %2.2f' % geometry.theta_step)
    