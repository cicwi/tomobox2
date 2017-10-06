#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains routines related to the data storage for the tomobox.
"""

import gc
import numpy

from PIL import Image      # image reader / writer
import os                  # file-name routines
import re                  # used in read_image_stack
from stat import ST_CTIME  # used in sort_by_date

from misc import *

# **************************************************************
#           image_array class
# **************************************************************

class data_blocks(object):
    """
    This will be a generic data array stored in RAM. Other types of arrays
    will be inherited from it (GPU, SSD etc.).
    
    User can get a block of data from the data_blocks by iterating it. 
    Or use 'total' property to set/get all data at once.
    """

    # Maximum block size in GB. 
    block_size = 1
    
    # Da data:
    _array = []

    # Block counter:
    _index = 0
    _block_step = 1
    _dim = 0

    def __init__(self, array = [], block_size = 4):
        
        self.block_size = block_size
        self.total = numpy.array(array, dtype = 'float32')
        
        print('Hello data_blocks!')
        
    def __del__(self): 
        
        del self._array
        gc.collect()
        
        print('Bye bye data_blocks!')
                    
    def __iter__(self):
        """
        Return itarator for the array.
        """
        
        # Reset the counter:
        self._index = 0
        self._dim = 0
        
        print('Accesing block data (%uGB block size)...' % self.block_size)  
        
        return self
        
    def _get_index(self, key):
        """
        Translate key into array index depending on the current block number.
        """
        block_step = self.length // self.block_number
        
        start = key * block_step
        stop = (key + 1) * block_step
        
        # Check if we are still inside the array:
        if start >= self.length:
            raise IndexError
            
        stop = min((stop, self.length)) 
        
        return start, stop
     
    def empty_block(self, val = 0):
        """
        Make a block of zeroes.
        """
        
        shp = self._array.shape
        shp[self._dim] /= self.block_number
        
        if val != 0:
            return numpy.zeros(shp)    
        else:
            return numpy.zeros(shp) + val    
        
    def empty_slice(self, val = 0):
        """
        Make a slice of zeroes.
        """
        
        shp = self._array.shape
        
        index = numpy.arange(2)
        index = index[index != self._dim]
        
        if val != 0:
            return numpy.zeros(shp[index])
        else:
            return numpy.zeros(shp[index]) + val

    def get_slice(self, key):
        """
        Get one slice.
        """        
        if self._dim == 0:        
            return self._array[key, :, :] 

        elif self._dim == 1:
             return self._array[:, key, :] 

        else:
            return self._array[:, :, key] 

    def set_slice(self, key, image):
        """
        Set one slice.
        """
        
        if self._dim == 0:        
            self._array[key, :, :] = image
        elif self._dim == 1:
             self._array[:, key, :] = image
        else:
            self._array[:, :, key] = image

    def __getitem__(self, key):
        """
        Get block of data.
        """
        
        #print("\r Block iterator access {:2.1%}".format((self._index) / self.block_number), end=" ")    
        
        progress_bar(key / self.block_number)
        
        start, stop = self._get_index(key)   

        if self._dim == 0:        
            return self._array[start:stop, :, :] 

        elif self._dim == 1:
             return self._array[:, start:stop, :] 

        else:
            return self._array[:, :, start:stop] 
        
    def __setitem__(self, key, data):    
        """
        Set block of data.
        """
        start, stop = self._get_index(key)   

        if self._dim == 0:        
            self._array[start:stop, :, :] = data

        elif self._dim == 1:
             self._array[:, start:stop, :] = data

        else:
            self._array[:, :, start:stop] = data

    def __len__(self):
        """
        Number of blocks.
        """
        return self.block_number

    '''
    def __next__(self):
        """
        Retrun next block of data.
        """
        
        try: 
            block = self[self._index]
            self._index += 1
            
        except IndexError:
            raise StopIteration
        
        print("\r Block iterator progress {:2.1%}".format((self._index) / self.block_number), end=" ")    
            
        return block 
    '''
    
    def block_index(self, key = []):
        """
        Return current block indexes. Can be used to create a meshgrid for instance.
        """

        # Current block:
        if key is []: key = self._index
        block = self[key]

        x = numpy.arange(0, block.shape[0])
        y = numpy.arange(0, block.shape[1])
        z = numpy.arange(0, block.shape[2])

        start, stop = self._get_index(key)   
        
        if self.dim == 0:
            x += start
        elif self.dim == 1:
            y += start
        else:
            z += start        

        return x,y,z
   
    @property    
    def total(self):
        return self._array
        
    @total.setter
    def total(self, array):
        self._array = array   
        
        
    @property
    def shape(self):
        """
        Dimensions of the array.
        """
        return self._array.shape
        
    @property
    def length(self):
        """
        Length of the array along the blocking direction.
        """
        return self.shape[self._dim]  

    @property
    def block_number(self):
        """
        Number of blocks.
        """
        return int(numpy.ceil(self.sizeGB / self.block_size))
        
    @property
    def size(self):
        """
        Number of elements of the array.
        """
        return self._array.size    
        
    @property
    def dtype(self):
        """
        Data type of the array.
        """
        return self._array.dtype
        
    @property
    def sizeGB(self):
        """
        Size of the array in Gigabytes
        """
        return (self.dtype.itemsize * self.size) / 1073741824
                
# **************************************************************
#           IO class
# **************************************************************
class io(object):
    """
    Reads / writes stacks of images.
    """
    
    _parent = None
    
    @staticmethod
    def _read_image(path_file):
        
        im = Image.open(path_file)
        return numpy.flipud(numpy.array(im, dtype = 'float32'))

    @staticmethod        
    def _sort_by_date(files):
        """
        Sort file entries by date
        """
        # get all entries in the directory w/ stats
        entries = [(os.stat(path)[ST_CTIME], path) for path in files]
    
        return [path for date, path in sorted(entries)]

    @staticmethod                
    def _sort_natural(files):
        """
        Sort file entries using the natural (human) sorting
        """
        # Keys
        keys = [int(re.findall('\d+', f)[-1]) for f in files]
    
        # Sort files using keys:
        files = [f for (k, f) in sorted(zip(keys, files))]
    
        return files
        
    @staticmethod
    def _read_image_stack(file, index_step = 1):
        """
        Low level image stack reader.
        """
        # Remove the extention and the last few letters:
        name = os.path.basename(file)
        ext = os.path.splitext(name)[1]
        name = os.path.splitext(name)[0]
        digits = len(re.findall('\d+$', name)[0])
        name_nonb = re.sub('\d+$', '', name)
        path = os.path.dirname(file)
        
        # Get the files of the same extension that finish by the same amount of numbers:
        files = os.listdir(path)
        files = [x for x in files if (re.findall('\d+$', os.path.splitext(x)[0]) and len(re.findall('\d+$', os.path.splitext(x)[0])[0]) == digits)]
    
        # Get the files that are alike and sort:
        files = [os.path.join(path,x) for x in files if ((name_nonb in x) and (os.path.splitext(x)[-1] == ext))]
        
        #files = sorted(files)
        files = io._sort_natural(files)
    
        #print(files)
    
        # Read the first file:
        image = io._read_image(files[0])
        sz = numpy.shape(image)
        
        file_n = len(files)//index_step
        
        data = numpy.zeros((file_n, sz[0], sz[1]), dtype = numpy.float32)
        
        # Read all files:
        for ii in range(file_n):
            
            filename = files[ii*index_step]
            try:
                a = io._read_image(filename)
            except:
                print('WARNING! FILE IS CORRUPTED. CREATING A BLANCK IMAGE.')
                a = numpy.zeros(data.shape[1:])
                
            if a.ndim > 2:
              a = a.mean(2)
              
            data[ii, :, :] = a
    
            print("\r Progress {:2.1%}".format((ii+1) / numpy.shape(files)[0] * index_step), end=" ")        
            
    
        print(ii, 'files were loaded.')
    
        return data
    
    @staticmethod
    def _read_images(path_folder, filter = [], x_roi = [], y_roi = [], index_range = [], index_step = 1):  
        """
        Find images in the folder, read them all!
        
        Args:
            filter (str): search for filenames caontaining the given string
            x_roi = [from, to]: horizontal range
            y_roi = [from, to]: vertical range
            index_range = [from, to]: read images in the given range
            index_step (int): step length in the reader loop
        """
        
        # if it's a file, read all alike, if a directory find a file to start from:
        if os.path.isfile(path_folder):
            filename = os.path.basename(path_folder)
            path = os.path.dirname(path_folder)

            # if file name is provided, the range is needed:
            if index_range != []:    
                first = index_range[0]
                last = index_range[1]

        else:
            # Try to find how many files do we need to read:

            # Get the files only:
            files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]

            # Get the last 4 letters:
            index = [os.path.splitext(x)[0][-4:] for x in files]

            # Filter out non-numbers:
            index = [int(re.findall('\d+', x)[0]) for x in index if re.findall('\d+', x)]

            # Extract a number from the first element of the list:
            first = min(index)

            # Extract a number from the first element of the list:
            last = max(index)

            print('We found projections in the range from ', first, 'to ', last, flush=True)

            # Find the file with the maximum index value:
            filename = [x for x in os.listdir(path) if str(last) in x][0]

            # Find the file with the minimum index:
            filename = sorted([x for x in os.listdir(path) if (filename[:-8] in x)&(filename[-3:] in x)])[0]

            print('Reading a stack of images')
            print('Seed file name is:', filename)
            if index_step > 0:
                print('Reading every %d-nd(rd) image.' % index_step)
            #srt = self.settings['sort_by_date']AMT24-25-SU1/

            data = io._read_image_stack(os.path.join(path,filename), index_step)
            
            if (index_range != []):
                data = data[index_range[0]:index_range[1], :, :]

            if (y_roi != []):
                data = data[:, y_roi[0]:y_roi[1], :]
            if (x_roi != []):
                data = data[:, :, x_roi[0]:x_roi[1]]

            return data
        
    @staticmethod    
    def _write_image(path_file, image):
        
        im = Image.fromarray(numpy.flipud(image))
        im.save(path_file)
        
    def read_dark(self, path_file):
        """
        Read the reference flat field image or several of them.
        """
        dark = self._read_image(path_file)
        dark = numpy.flipud(dark)
        
        self._parent._dark = dark
            
        # add record to the history:
        self._parent.meta.history.add_record('io.read_dark', path_file)
        print('Flat field reference image loaded.')
    
    def read_ref(self, path_files):
        """
        Read reference flat field. Can specify a single file path or an array with several files.
        """
        
        ref = []

        if type(path_files) == str:
            ref  = self._read_image(path_files)
            
        elif type(path_files) == list:
            for file in path_files:
                if os.path.isfile(file): ref.append(self._read_image(file))
                
        else:
            print('path_files parameter in read_ref() should be iether a full file name or a list of file names')
            
        # Swap the axses for ASTRA:
        ref = numpy.transpose(ref, [1,0,2])    

        self._parent._ref = ref
       
        # add record to the history:
        self._parent.meta.history.add_record('io.read_ref', path_files)           
        self._parent.message('Flat field reference image loaded.')
           
    def read_projections(self, path = '', filter = ''):
        '''
        Read projection data.
        '''        
        data = self._read_images(path, filter)
                    
        # Transpose to satisfy ASTRA dimensions if loading projection data:    
        self._parent.data.total = numpy.transpose(data, (1, 0, 2))
        
        # Collect garbage:
        gc.collect()
        
        # add record to the history:
        self._parent.meta.history.add_record('io.read_projections', path)
        
    def read_slices(self, path = '', filter = ''):
        '''
        Read volume slices.
        '''        
        data = self._read_images(path, filter) 
        
        self._parent.data.total = data
        
        # Collect garbage:
        gc.collect()
        
        # add record to the history:
        self._parent.meta.history.add_record('io.read_slices', path)
            
    def read_flexray(self, path):
        '''
        Read raw projecitions, dark and flat-field, scan parameters,
        '''
        self.read_dark(path + '/di0000.tif')
        
        self.read_ref([path + '/io0000.tif', path + '/io0001.tif'])
        
        self.read_projections(path)
        
        self.parse_flexray(path)
        
    
# **************************************************************
#           PROJECTIONS class
# **************************************************************

class projections(object):
    """
    Container for the projection data.
    """

    _data = data_blocks()
    _ref  = []
    _dark = []
    
    def __init__(self):
        pass
    
    def __del__(self):
        pass
    
    def get_vector ->>> to data block
    
    def get_ref(self, proj_num = 0):
        '''
        Returns a reference image. Interpolated if sefveral reference images are available.
        '''
        
        # Return reference image for the current projection:
        if self._ref.ndim > 2:
          
            if self._data is None:
                self._parent.warning('No raw data available. We don`t know how many projections there are in order to interpolate the reference image properly. Read raw data first.')
                dsz = self._ref.shape[1]

            else:
                dsz = self.data.shape[1]
                
            # Several flat field images are available:
            ref = self._ref
          
            sz = ref.shape

            # This implementation is too slow:
            #proj_index = numpy.linspace(0, dsz-1, sz[1])
            #interp_grid  = numpy.array(numpy.meshgrid(numpy.arange(sz[0]), proj_num, numpy.arange(sz[2])))
            #interp_grid = numpy.transpose(interp_grid, (2,1,3,0))
            #original_grid = (numpy.arange(sz[0]), proj_index, numpy.arange(sz[2]))
            #return interp_sc.interpn(original_grid, ref, interp_grid)
            
            proj_index = numpy.linspace(0, sz[1]-1, dsz)
            
            a = proj_index[proj_num]
            fract = a - numpy.floor(a)            
            a = int(numpy.floor(a))            
            
            if a < (dsz-1):
                b = int(numpy.ceil(proj_index[proj_num]))
            else:
                b = a
                
            return self._ref[:, a, :] * (1 - fract) + self._ref[:, b, :] * fract
          
        else:
            
           # One flat field image is available:
           return self._ref  