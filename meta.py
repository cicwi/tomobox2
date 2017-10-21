#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains routines related to the meta-data handling for the tomobox.
"""

# External modules
import numpy
import time
import astra
import transforms3d

# Tomobox modules
import reports               # Makes compact reports for different classes.
import misc

# **************************************************************
#           HISTORY class
# **************************************************************   
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
    def list_operations(self):
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
class geometry(misc.subclass):
    '''
    Geometry class describes the simplest circular geometry. 
    src2obj, det2obj, det_pixel and thetas fully describe it. 
    
    Main task of the geometry class is to generate ASTRA vector that describes 
    coordinates of the detector and source relative to the volume.
    
    More complex geometries, flex_geometry, for instance, will be inherited.
    
    '''    
    
    # Private properties:
    _src2obj = 0
    _det2obj = 0
    _det_pixel = []
    _thetas = []

    # Methods:
    def __init__(self, parent, src2obj = 1, det2obj = 1, det_pixel = [1, 1], theta_range = [0, numpy.pi*2], theta_n = 2):
        '''
        Make sure that all relevant properties are set to some value.
        '''
        misc.subclass.__init__(self, parent)
        
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
        '''
        Source-to-object distance
        '''
        return self._src2obj
        
    @src2obj.setter
    def src2obj(self, src2obj):
        '''
        Source-to-object distance
        '''
        self._src2obj = src2obj
        
    @property
    def det2obj(self):
        '''
        Detector-to-object distance
        '''
        return self._det2obj
        
    @det2obj.setter
    def det2obj(self, det2obj):
        '''
        Detector-to-object distance
        '''
        self._det2obj = det2obj
        
    @property
    def magnification(self):
        '''
        Magnification. Computed assuming image plane at the rotation axis.
        '''
        return (self._det2obj + self._src2obj) / self._src2obj
        
    @property
    def src2det(self):
        '''
        Source-to-detector distance
        '''
        return self._src2obj + self._det2obj
        
    @property
    def det_pixel(self):
        '''
        Physical detector pixel size.
        '''
        return self._det_pixel
        
    @det_pixel.setter
    def det_pixel(self, det_pixel):
        '''
        Physical detector pixel size.
        '''
        self._det_pixel = det_pixel
        
    @property
    def img_pixel(self):
        '''
        Backprojection pixel size taking into account magnification.
        '''
        return [self._det_pixel[1] / self.magnification, self._det_pixel[1] / self.magnification, self._det_pixel[0] / self.magnification]  
        
    @img_pixel.setter
    def img_pixel(self, img_pixel):
        '''
        Backprojection pixel size taking into account magnification.
        '''
        self._det_pixel = [img_pixel[0] * self.magnification, img_pixel[1] * self.magnification]        
        
    @property
    def det_shape(self):
        '''
        Get the detector shape in pixels
        '''
        if self._parent.data is None:
            self._parent.warning('No raw data in the pipeline. The detector size is not known.')
        else:
            return self._parent.data.slice_shape
            
    @property
    def det_size(self):
        '''
        Get the detector shape in mm
        '''
        
        # We wont take into account the det_size from the log file. Only use actual data size.
        if self._parent.data is None:
            self._parent.warning('No raw data in the pipeline. The detector size is not known.')
        else:
            return self._det_pixel * self._parent.data.slice_shape
                
    @property
    def thetas(self):
        '''
        Theats of the whole dataset.
        '''
        # Check consistency with the data size:
        if not self._parent.data is None:
            if self._parent.data.length != numpy.size(self._thetas):
                self._parent.message('Assuming uniform angular sampling. Initializing thetas using the data shape.')
            
                self.init_thetas(theta_n = self._parent.data.length)
            
        return numpy.array(self._thetas)
        
    @thetas.setter
    def thetas(self, thetas):
        '''
        Set thetas of the whole dataset
        '''
        self._thetas = numpy.array(thetas)
    
    @property
    def block_thetas(self):
        '''
        Return thetas of the current data block.
        '''
        
        thetas = self.thetas
        index = slice(self._parent.data._index)
        
        return thetas[index]
    
    @property
    def theta_n(self):
        """
        Number of projections.
        """
        return self.thetas.size    
        
    @property
    def theta_range(self):
        """
        First to last angle.
        """
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
            
    def get_astra_vector(self, blocks = False):
        """
        Generate the vector that describes positions of the source and detector. + Returns the corrsponding projection geometry.
        If block == True: returns geometry of the current block.
        """
        
        sz = self._parent.data.slice_shape
        det_count_x = sz[1]
        det_count_z = sz[0]

        # Inintialize ASTRA projection geometry to import vector from it
        if blocks:
            thetas = self.thetas[self._parent.data.block_index]
            
        else:
            thetas = self.thetas
            
        proj_geom = astra.create_proj_geom('cone', self.det_pixel[1], self.det_pixel[0], det_count_z, det_count_x, thetas, self.src2obj, self.det2obj)
        proj_geom = astra.functions.geom_2vec(proj_geom)
        
        return proj_geom['Vectors'], proj_geom

    def get_proj_geom(self, blocks = False):
        """
        Generate ASTRA progection geometry.
        """
        vect, geom = self.get_astra_vector(blocks)
        
        return geom
   
# **************************************************************
#           VOLUME_GEOMETRY class
# **************************************************************        
class vol_geometry(misc.subclass):
    """
    A separate compact class to describe volume geometry.
    """
    _vol_trans = [0, 0, 0]     # not per projection
    _vol_rot = [0, 0, 0]       # not per projecrion
    
    _img_pixel = [0, 0, 0]     # voxel size in mm
    
    def __init__(self, parent):
        
        misc.subclass.__init__(self, parent)
        
    @property
    def vol_shape(self):
        '''
        Get the total volume shape in pixels
        '''
        if self._parent.data is None:
            self._parent.warning('Volume data is not initialized. The volume shape is [0, 0, 0].')
            return numpy.zeros(3)
            
        else:
            return self._parent.data.shape
        
    @property
    def vol_size(self):
        '''
        Get the total volume size in mm
        '''
        
        shape = self.vol_shape
        
        return self.img_pixel * shape
            
    @property
    def img_pixel(self):
        '''
        Voxel size in mm.
        '''
        return numpy.array(self._img_pixel)
        
    @img_pixel.setter    
    def img_pixel(self, img_pixel):
        '''
        Voxel size in mm.
        '''        
        self._img_pixel = img_pixel
                
    def rotate(self, euler):
        """
        Rotate the system of coordinates of the reconstruction volume using Euler angles
        """
        self._vol_rot += euler
    
    def translate(self, vector):
        """
        Translate the system of coordinates of the reconstruction volume by a vector
        """
        self._vol_trans += vector
        
    def modify_proj_geom(self, proj_geom, blocks = False):
        '''
        Use this to apply rotation and translation of the volume to the projection geometry that describes projection data.
        '''
        
        # Extract vector:
        proj_geom = astra.functions.geom_2vec(proj_geom)
        vectors = proj_geom['Vectors']
    
        # Extract relevant vectors from the total ASTRA vector:
        #det_shape = geometry.det_shape
        det_shape = [proj_geom['DetectorRowCount'], proj_geom['DetectorColCount']]
        px = [proj_geom['DetectorSpacingY'], proj_geom['DetectorSpacingX']]
              
        src_vect = vectors[:, 0:3]
        det_vect = vectors[:, 3:6]
        det_axis_vrt = vectors[:, 9:12] * det_shape[0] / 2
        det_axis_hrz = vectors[:, 6:9] * det_shape[1] / 2
       
        # Global transformation:
        # Rotation matrix based on Euler angles:
        if self.vol_rot.abs().sum() > 0:
            R = transforms3d.euler.euler2mat(self.vol_rot, 'szxy')
    
            # Apply transformation:
            det_axis_hrz[:] = numpy.dot(R, det_axis_hrz)
            det_axis_vrt[:] = numpy.dot(R, det_axis_vrt)
            src_vect[:] = numpy.dot(R, src_vect)
            det_vect[:] = numpy.dot(R, det_vect)            
            
        # Add translation:
        if self.vol_trans.abs().sum() > 0:    
            vect_norm = det_axis_vrt[2]
                
            T = numpy.array([self.vol_trans[0] * vect_norm / px[1], self.vol_trans[2] * vect_norm / px[1], self.vol_trans[1] * vect_norm / px[0]])    
            src_vect[:] -= T            
            det_vect[:] -= T
            
        # Update proj_geom:
        sz = self._parent.data.slice_shape
        
        det_count_x = sz[1]
        det_count_z = sz[0]
            
        return astra.create_proj_geom('cone_vec', det_count_z, det_count_x, vectors)    
             
    def get_vol_geom(self, blocks = False):
        '''
        Initialize volume geometry.        
        '''
        
        # Shape and size (mm) of the volume
        shape = self.vol_shape
        size = shape * self.img_pixel
        
        if blocks:
            # Generate volume geometry for one chunk of data:
            start = self._parent.data._index[0]    
            stop = self._parent.data._index[-1]
            
            length = self._parent.data.length
            
            # Compute offset from the centre:
            centre = (length - 1) / 2
            offset = (start + stop) / 2 - centre
            
            shape[1] = (stop - start + 1) 
            size = shape * self.img_pixel
        
        else:
            offset = 0
            
        vol_geom = astra.create_vol_geom(shape[1], shape[0], shape[2], 
                  -size[0]/2 + offset, size[0]/2+ offset, -size[1]/2, size[1]/2, 
                  -size[2]/2, size[2]/2)
            
        return vol_geom
    
# **************************************************************
#           FLEX_GEOMETRY class
# **************************************************************          
class flex_geometry(geometry):
    """
    This class describes the circular geometry in the Flexray scanner, 
    where the source and the detector can be moved to some initial positions prior to the circular scan. 
    
    All properties are given in "mm" and describe changes relatively to a default circular orbit.
    """

    # Deviations from the standard circular geometry:
    # unit: mm
    # orientation: [horizontal, vertical, magnification]
    det_trans = [0, 0, 0]
    det_rot = 0
    
    src_trans = [0, 0, 0]
    
    axs_trans = [0, 0, 0]
    
    vol_trans = [0, 0, 0]     # not per projection
    vol_rot = [0, 0, 0]       # not per projecrion
    
    theta_offset = []
    
    '''
    Modifiers (dictionary of geometry modifiers that can be applied globaly or per projection)
    VRT, HRZ and MAG are vertical, horizontal and prependicular directions relative to the original detector orientation    
    '''
    def _unit2mm(self, unit):
        """
        Convert some units into mm.
        """
        units = {'pixel': [self.det_pixel[1], self.det_pixel[1], self.det_pixel[0]], 
            'mm': [1, 1, 1], 'um': [1e-3, 1e-3, 1e-3], 'micron': [1e-3, 1e-3, 1e-3]}
        
        if unit in units:
            return numpy.array(units[unit])
        else:
            raise ValueError('Unit not recognized!')
                
    def set_volume_shift(self, shift, unit = 'pixel', additive = True):
        """
        Translate reconstruction volume relative to the rotation center.
        """
        if additive:
            self.vol_trans += shift * self._unit2mm(unit)
        else:
            self.vol_trans = shift * self._unit2mm(unit) 
        
    def set_thermal_shift(self, thermal_shifts, unit = 'pixel', additive = False):
        '''
        Shift the source according to the thermal shift data. Thermal shift is in pixels
        '''
        unt = self._unit2mm(unit)
        
        if additive:
            self.src_trans[0] += thermal_shifts[:,0]/(self.magnification - 1) * unt[0]
            self.src_trans[1] += thermal_shifts[:,1]/(self.magnification - 1) * unt[1]
        else:
            self.src_trans[0] = thermal_shifts[:,0]/(self.magnification - 1) * unt[0]
            self.src_trans[1] = thermal_shifts[:,1]/(self.magnification - 1) * unt[1]

    def set_rotation_axis(self, shift, unit = 'pixel', additive = False):
        '''
        shift is in pixels.
        '''
        
        unt = self._unit2mm(unit)
        
        if additive:
            self.axs_trans[0] += shift / self.magnification * unt[0]
        else:
            self.axs_trans[0] = shift / self.magnification * unt[0]

    def get_rotation_axis(self, unit = 'pixel'):
        '''
        Retrurn the rotation axis shift in pixels.
        '''
        
        unt = self._unit2mm(unit)
        
        return self.axs_trans[0] * self.magnification / unt[0]

    def set_optical_axis(self, shift, unit = 'pixel', additive = False):
        '''
        Set vertical projected position of the source relative to the center of the detector.
        '''
        
        unt = self._unit2mm(unit)
        
        if additive:
            self.det_trans[1] += shift * unt[0]
        else:
            self.det_trans[1] = shift * unt[0]

        # Center the volume around the new axis:
        self.shift_volume([0, shift, 0])
                
    def get_volume_offset(self, unit = 'pixel'):
        '''
        Compute the shift of the volume central point [x, y, z] due to offsets in the optical and rotation axes.
        '''
        unt = self._unit2mm(unit)
        
        # In case traslations are provided per projection, find averages
        det_hrz = numpy.mean(self.det_trans[0])
        src_hrz = numpy.mean(self.det_trans[0])
        det_vrt = numpy.mean(self.det_trans[1])
        src_vrt = numpy.mean(self.det_trans[1])
        axs_hrz = numpy.mean(self.axs_trans[0])
        
        hrz = 3 * (abs(det_hrz * self.src2obj + src_hrz * self.det2obj) / self.src2det - axs_hrz) / unt[0]
        vrt = abs(det_vrt * self.src2obj + src_vrt * self.det2obj) / self.src2det - self.vol_trans[1] / unt[1]
        
        return [hrz, vrt]

    def get_pixel_coords(self):
        '''
        Generate pixel coordinates of the current tile
        '''
        x0 = self.det_trans[0]
        y0 = self.det_trans[1]

        sz_x = self.det_size[1]
        sz_y = self.det_size[0]

        xx = numpy.linspace(0, sz_x, self.det_shape[1]) + x0
        yy = numpy.linspace(0, sz_y, self.det_shape[1]) + y0
                            
        return xx, yy
        
    @staticmethod
    def _per_projection(vector, index):
        """
        If vector is 1D - return vector, if it is given for every theta - return one that has the index = index.
        """
        if numpy.ndim(vector) < 2:
            return vector
        else:
            return vector[:, index]
      
    def get_detector_vector(self):
        """
        Return coordinates of the detector boundary for each projection:
        [left, right], [top, bottom]
        """
        vectors = self.get_astra_vector()
        
        det_centre = vectors[:, 3:6].copy() 
        det_shape = geometry.det_shape
        
        det_axis_vrt = vectors[:, 9:12] * det_shape[0] / 2 
        det_axis_hrz = vectors[:, 6:9] * det_shape[1] / 2
       
        det_top = det_centre + det_axis_vrt
        det_bottom = det_centre - det_axis_vrt

        det_left = det_centre + det_axis_hrz
        det_right = det_centre - det_axis_hrz

        return [det_left, det_right], [det_top, det_bottom]

    def get_source_vector(self):
        """
        Return coordinates of the source for each projection.
        """
        vectors = self.get_astra_vector()
        
        src_vect = vectors[:, 0:3].copy() 

        return src_vect
        
    def get_volume_parameters(self):
        """
        Compute volume size correcponding to the currecnt geometry.
        """
        
        # Source positions:
        src = self.get_source_vector()
        
        # Left\right and top/bottom of the detector:
        lr, tb = self.get_detector_vector()
        
        print(src)
        print(lr[0])      
  
    def get_astra_vector(self, blocks):
        """
        Generate the vector that describes positions of the source and detector.
        """
        # Call parent method
        vectors, proj_geom = geometry.get_astra_vector(self, blocks)
        
        # Modify vector and apply it to astra projection geometry:
        for ii in range(0, vectors.shape[0]):
            
            # Define vectors:
            src_vect = vectors[ii, 0:3]    
            det_vect = vectors[ii, 3:6]    
            det_axis_hrz = vectors[ii, 6:9]          
            det_axis_vrt = vectors[ii, 9:12]

            #Precalculate vector perpendicular to the detector plane:
            det_normal = numpy.cross(det_axis_hrz, det_axis_vrt)
            det_normal = det_normal / numpy.sqrt(numpy.dot(det_normal, det_normal))
            
            # Translations relative to the detecotor plane:
            px = self.det_pixel
                
            #Detector shift (V):
            det_vect += self._per_projection(self.det_trans[1], ii) * det_axis_vrt / px[0]
    
            #Detector shift (H):
            det_vect += self._per_projection(self.det_trans[0], ii) * det_axis_hrz / px[1]
    
            #Detector shift (M):
            det_vect += self._per_projection(self.det_trans[2], ii) * det_normal /  px[1]
    
            #Source shift (V):
            src_vect += self._per_projection(self.src_trans[1], ii) * det_axis_vrt / px[0]  
    
            #Source shift (H):
            src_vect += self._per_projection(self.src_trans[0], ii) * det_axis_hrz / px[1]

            #Source shift (M):
            src_vect += self._per_projection(self.src_trans[2], ii) * det_normal / px[1]

            # Rotation axis shift:
            det_vect -= self._per_projection(self.axs_trans[0], ii) * det_axis_hrz  / px[1]
            src_vect -= self._per_projection(self.axs_trans[0], ii) * det_axis_hrz  / px[1]
    
            # Rotation relative to the detector plane:
            # Compute rotation matrix
        
            T = transforms3d.axangles.axangle2mat(det_normal, self.det_rot)
            
            det_axis_hrz[:] = numpy.dot(T.T, det_axis_hrz)
            det_axis_vrt[:] = numpy.dot(T, det_axis_vrt)
        
            # Global transformation:
            # Rotation matrix based on Euler angles:
            R = transforms3d.euler.euler2mat(self.vol_rot[0], self.vol_rot[1], self.vol_rot[2], 'szxy')
    
            # Apply transformation:
            det_axis_hrz[:] = numpy.dot(R, det_axis_hrz)
            det_axis_vrt[:] = numpy.dot(R, det_axis_vrt)
            src_vect[:] = numpy.dot(R, src_vect)
            det_vect[:] = numpy.dot(R, det_vect)            
            
            # Add translation:
            vect_norm = det_axis_vrt[2]
            
            T = numpy.array([self.vol_trans[0] * vect_norm / px[1], self.vol_trans[2] * vect_norm / px[1], self.vol_trans[1] * vect_norm / px[0]])    
            src_vect[:] -= T            
            det_vect[:] -= T
        
        # Update proj_geom:
        sz = self._parent.data.slice_shape
        det_count_x = sz[1]
        det_count_z = sz[0]
            
        proj_geom = astra.create_proj_geom('cone_vec', det_count_z, det_count_x, vectors)    
        
        return vectors, proj_geom

# **************************************************************
#           PROJ_META class
# **************************************************************        
class proj_meta(misc.subclass):
    '''
    This object contains various properties of the imaging system and the history of pre-processing.
    '''
    geometry = None
    history = history()

    def __init__(self, parent):
        misc.subclass.__init__(self, parent)
        
        self.geometry = flex_geometry(parent)
        #self.geometry = geometry(parent)
        
    physics = {'voltage': 0, 'current':0, 'exposure': 0}
    lyrics = {}
    
# **************************************************************
#           VOL_META class
# **************************************************************        
class vol_meta(misc.subclass):
    '''
    This object contains properties of the volume system and the history of pre-processing.
    '''
    history = history()  
    geometry = None
    
    def __init__(self, parent):
        misc.subclass.__init__(self, parent)
        
        self.geometry = vol_geometry(parent)