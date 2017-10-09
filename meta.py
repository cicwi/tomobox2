#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains routines related to the meta-data handling for the tomobox.
"""

import reports               # Makes compact reports for different classes.

class meta(subclass):
    '''
    This object contains various properties of the imaging system and the history of pre-processing.
    '''
    geometry = None
    history = history()

    def __init__(self, parent):
        subclass.__init__(self, parent)
        
        self.geometry = geometry(self._parent)
        
    physics = {'voltage': 0, 'current':0, 'exposure': 0}
    lyrics = {}
    
class history():
    '''
    Container for the history reconrds of operations applied to the data.
    '''
    
    _records = []
    
    @property
    def records(self):
        return self._records.copy()
        
    def __init__(self):
        
        self._records = []
        self.add_record('Created')
    
    def add_record(self, operation = '', properties = None):
        
        # Add a new history record:
        timestamp = time.ctime()
        
        # Add new record:
        self._records.append([operation, properties, timestamp, numpy.shape(self._records)[0] ])
    
    @property
    def keys(self):
        '''
        Return the list of operations.
        '''
        return [ii[0] for ii in self._records]
        
    def find_record(self, operation):
        # Find the last record by its operation name:
            
        result = None
        
        result = [ii for ii in self._records if operation in ii[0]]

        if numpy.size(result) > 0:
            return result[-1]    
        else:
            return None
    
    def time_last(self):
        # Time between two latest records:
        if self._records.size() > 0:
            return self._records[-1][2] - self._records[-2][2]
            
    def _delete_after(self, operation):
        # Delete the history after the last backup record:
            
        record = self.find_record(self, operation)
        
        # Delete all records after the one that was found:
        if not record is None:
            self._records = self._records[0:record[3]]
    
# **************************************************************
#           Geometry class
# **************************************************************
class geometry(subclass):
    '''
    Geometry class describes the simplest circular geometry. 
    src2obj, det2obj, det_pixel and thetas fully describe it. 
    
    Main task of the geometry class is to generate ASTRA vector that describes 
    coordinates of the detector and source relative to the volume.
    
    More complex geometries, flex_geometry, for instance, will be inherited.
    
    '''    
    
    # Private properties:
    _src2obj
    _det2obj
    _det_pixel
    _thetas

    # Methods:
    def __init__(self, parent, src2obj = 1, det2obj = 1, det_pixel = [1, 1], theta_range = [0, numpy.pi*2], theta_n = 2):
        '''
        Make sure that all relevant properties are set to some value.
        '''
        subclass.__init__(self, parent)
        
        self._src2obj = src2obj
        self._det2obj = det2obj
        self._det_pixel = det_pixel
        self.init_thetas(theta_range, theta_n)
    
    def report(self):
        '''
        Print a report of the geometry.
        '''
        reports.report_geometry(self)
        
    # Set/Get methods (very bodring part of code but, hopefully, it will make geometry look prettier from outside):              
    @property
    def src2obj(self):
        return self._src2obj
        
    @src2obj.setter
    def src2obj(self, src2obj):
        self._src2obj = src2obj
        
    @property
    def det2obj(self):
        return self._det2obj
        
    @det2obj.setter
    def det2obj(self, det2obj):
        self._det2obj = det2obj
        
    @property
    def magnification(self):
        return (self._det2obj + self._src2obj) / self._src2obj
        
    @property
    def src2det(self):
        return self._src2obj + self._det2obj
        
    @property
    def det_pixel(self):
        return self._det_pixel
        
    @det_pixel.setter
    def det_pixel(self, det_pixel):
        self._det_pixel = det_pixel
        
    @property
    def img_pixel(self):
        return [self._det_pixel[0] / self.magnification, self._det_pixel[1] / self.magnification]  
        
    @img_pixel.setter
    def img_pixel(self, img_pixel):
        self._det_pixel = [img_pixel[0] * self.magnification, img_pixel[1] * self.magnification]        
        
    @property
    def det_size(self):
        
        # We wont take into account the det_size from the log file. Only use actual data size.
        if self._parent.data is None:
            self._parent.warning('No raw data in the pipeline. The detector size is not known.')
        else:
            return self._det_pixel * self._parent.data.slice_shape
        
    @property
    def thetas(self):
        data = self._parent.data.
        dt = data.sampling[data.dim]
        
        # Check consistency with the data size:
        if not self._parent.data is None:
            if self._parent.data.length != numpy.size(self._thetas[::dt]):
                self._parent.message('Assuming uniform angular sampling. Initializing thetas using the data shape.')
            
                self.init_thetas(theta_n = self._parent.data.length)
            
        return numpy.array(self._thetas[::dt])
        
    @thetas.setter
    def thetas(self, thetas):
        # Doesn't take into account the sampling!
        self._thetas = numpy.array(thetas)
    
    @property
    def theta_n(self):
        return self.thetas.size    
        
    @property
    def theta_range(self):
        if self.thetas.size > 0:
            return (self.thetas[0], self.thetas[-1])
        else:
            return (0, 0)
    
    @theta_range.setter    
    def theta_range(self, theta_range):
        # Change the theta range:
        len = numpy.size(self.thetas)    
        
        if len > 2:
            self.thetas = numpy.linspace(theta_range[0], theta_range[1], len)
        else:
            self.thetas = [theta_range[0], theta_range[1]]
        
    @property        
    def theta_step(self):
        return numpy.mean(self._thetas[1:] - self._thetas[0:-1])  
        
    def init_thetas(self, theta_range = [], theta_n = 2):
        # Initialize thetas array. You can first initialize with theta_range, and add theta_n later.
        if theta_range == []:
            self._thetas = numpy.linspace(self._thetas[0], self._thetas[-1], theta_n)
        else:    
            self._thetas = numpy.linspace(theta_range[0], theta_range[1], theta_n)    
    
    # TODO:
    def get_vector(self):
        pass
    
class flex_geometry(geometry):
    """
    This class describes the circular geometry in the Flexray scanner, 
    where the source and the detector can be moved to some initial positions prior to the circular scan. 
    
    All properties are given in "mm" and describe changes relatively to a default circular orbit.
    """

    # Modifiers describe 
    det_trans = [0, 0, 0]
    det_rot = 0
    
    src_trans = [0, 0, 0]
    
    axs_trans = [0, 0, 0]
    
    vol_trans = [0, 0, 0]
    vol_rot = [0, 0, 0]
    
    theta_offset = 0
    
    '''
    Modifiers (dictionary of geometry modifiers that can be applied globaly or per projection)
    VRT, HRZ and MAG are vertical, horizontal and prependicular directions relative to the original detector orientation    
    '''
    @staticmethod
    def _unit2mm(unit):
        """
        Convert some units into mm.
        """
        units = {'pixel': [self.det_pixel[1], self.det_pixel[1], self.det_pixel[0]], 
        'mm': [1, 1, 1], 'um': [1e-3, 1e-3, 1e-3], :'micron': [1e-3, 1e-3, 1e-3]}
        
        if unit in units:
            return = numpy.array(units[unit])
        else:
            raise ValueError('Unit not recognized!')
                
    def shift_volume(self, shift, unit = 'pixel', additive = True):
        """
        Translate reconstruction volume relative to the rotation center.
        """
        if additive:
            self.vol_trans += shift * self.unit2mm(unit)
        else:
            self.vol_trans = shift * self.unit2mm(unit) 
        
    def thermal_shift(self, thermal_shifts, unit = 'pixel', additive = False):
        '''
        Shift the source according to the thermal shift data. Thermal shift is in pixels
        '''
        if additive:
            src_trans += thermal_shifts[:,0]/(self.magnification - 1) * self.det_pixel[1]
            self.modifiers['src_vrt'] += thermal_shifts[:,1]/(self.magnification - 1) * self.det_pixel[0]
        else:
            self.modifiers['src_hrz'] = thermal_shifts[:,0]/(self.magnification - 1) * self.det_pixel[1]
            self.modifiers['src_vrt'] = thermal_shifts[:,1]/(self.magnification - 1) * self.det_pixel[0]

    def set_rotation_axis(self, shift, additive = False):
        '''
        shift is in pixels.
        '''
        if additive:
            self.modifiers['axs_hrz'] += shift / self.magnification * self.det_pixel[1]
        else:
            self.modifiers['axs_hrz'] = shift / self.magnification * self.det_pixel[1]

    def get_rotation_axis(self):
        '''
        Retrurn the rotation axis shift in pixels.
        '''
        return self.modifiers['axs_hrz'] * self.magnification / self.det_pixel[1]

    def optical_axis_shift(self, shift, additive = False):
        if additive:
            self.modifiers['det_vrt'] += shift * self.det_pixel[0]
        else:
            self.modifiers['det_vrt'] = shift * self.det_pixel[0]

        # Center the volume around the new axis:
        self.translate_volume([0, 0, shift])
                
    def origin_shift(self):
        '''
        Compute the shift of the volume central point [x, y, z] due to offsets in the optical and rotation axes.
        '''    
        hrz = 3 * (abs(self.modifiers['det_hrz'] * self.src2obj + self.modifiers['src_hrz'] * self.det2obj) / self.src2det - self.modifiers['axs_hrz'])  
        #vrt = (self.modifiers['det_vrt'] * self.src2obj + self.modifiers['src_vrt'] * self.det2obj) / self.src2det
        vrt = abs(self.modifiers['det_vrt'] * self.src2obj + self.modifiers['src_vrt'] * self.det2obj) / self.src2det - self.modifiers['vol_z_tra']
        
        # Take into account global shifts:
        #hrz = numpy.max([numpy.abs(hrz + self.modifiers['vol_x_tra']), hrz + numpy.abs(self.modifiers['vol_y_tra'])])
        #vrt += self.modifiers['vol_z_tra']
        
        return [hrz / self.det_pixel[1], vrt / self.det_pixel[0]]
    
    
    
