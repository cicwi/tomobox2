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

import random
import misc

class ram_data_pool(object):
    """
    Data located in RAM memory
    """
    _data = None
    _dim = 0
    
    def __init__(self, dim):
        
        self._dim = dim
        
    def __del__(self):            
        
        self._data = None
        gc.collect()
        
    def write(self, key, image):        
        misc._set_dim_data(self._data, self._dim, key, image)
        
    def read(self, key):
        return misc._get_dim_data(self._data, self._dim, key)
    
    @property    
    def total(self):
        return self._data
        
    @total.setter    
    def total(self, total):
        self._data = total   
        
    @property
    def shape(self):    
        return numpy.array(self._data.shape)
        
    @property
    def dtype(self):
        return numpy.dtype(self._data.dtype)    
    
class swap_data_pool(object):
    """
    Data lockated on disk
    """    
    # Swap path:
    _swap_path = ''
    _swap_name = 'swap'
    
    _dim = 0
    _shape = None
    _dtype = 'float32'
    
    def __init__(self, swap_path, swap_name, dim, shape = None, dtype = 'float32'):
        '''
        
        '''
        self._swap_path = swap_path
        self._swap_name = swap_name        
        
        if not os.path.exists(self._swap_path):
            os.makedirs(self._swap_path)
            
        self._dim = dim
        self._shape = shape
        self._dtype = dtype 
            
    def __del__(self):        
        
        # Free swap:
        self._remove_swap()
        
    def write(self, key, image):
        self._write_swap(key, image)
        
    def read(self, key):
        return self._read_swap(key)
        
    def _read_swap(self, key):
        """
        Read a single swap image.
        """
        file = os.path.join(self._swap_path, self._swap_name + '_%05u.tiff' % key)
        
        # Read image:
        return io._read_image(file)
        
    def _write_swap(self, key, image):
        """
        Write a single swap image.
        """        
        file = os.path.join(self._swap_path, self._swap_name + '_%05u.tiff' % key)
        
        # Write image:
        io._write_image(file, image)
        
    def _remove_swap(self):
        """
        Removes all swap files.
        """
        path_files = os.path.join(self._swap_path, self._swap_name + '*.tiff')
        
        if os.path.exists(path_files):
            try:
                os.remove(path_files)
                print('Swap files removed.')
                
            except:
                print('Failed to remove swap files at: ' + path_files)            

    @property    
    def total(self):
        
        if self._shape is None:
            raise ValueError('Swap data pool shape is not known! Can initialize total data.')
            
        array = numpy.zeros(self._shape)
        
        for ii in range(self._shape[self._dim]):
            misc._set_dim_data(array, self._dim, ii, self._read_swap(ii))
            
        return array
        
    @total.setter    
    def total(self, array):
        
        self._shape = array.shape
        
        for ii in range(self._shape[self._dim]):
            image = misc._get_dim_data(array, self._dim, ii)
            
            self._write_swap(ii, image)
            
    @property
    def shape(self):    
        return numpy.array(self._shape)
        
    @property
    def dtype(self):
        return self._dtype            
            
# **************************************************************
#           DATA_BLOCKS class
# **************************************************************
class data_array(object):
    """
    This is a data array that contains information about the data and can set/get
    data in blocks or slices. It will request data from either the ram_data_pool
    or swap_data_pool.
    """    
    # Block buffer and it's current key:
    _block = []
    _block_key = -1
    _block_updated = False
    
    # Additional buffer for the slice image:
    _slice = []
    _slice_key = -1
    _slice_updated = False
    
    # Global index from 0 to theta_n - 1
    _global_index = []
   
    # Maximum block size in GB. 
    _block_sizeGB = 1
        
    # Where do we get the data from?
    _data_pool = None
    
    # Indexer, can be 'sequential', 'random', 'equidistant':
    _indexer = 'sequential'    
        
    
    def __init__(self, array = None, shape = None, dtype = 'float32', block_sizeGB = 1, dim = 1, swap = False):
        
        # initialize the data pool:
        if swap:
            self._data_pool = swap_data_pool('/export/scratch3/kostenko/Fast_Data/swap', 'swap', dim)
        else:
            self._data_pool = ram_data_pool(dim)
                        
        self._block_sizeGB = block_sizeGB
                        
        if array is not None:
            self._data_pool.total = numpy.array(array, dtype = dtype)            
            
        elif shape is not None:
            self._data_pool.total = numpy.zeros(shape, dtype = dtype)            
                    
    def __del__(self):
        
        # Double check that the last buffer was written on disk:
            
        self.finalize_slice()  
        self.finalize_block()  
        
        # Free memory:
        self._block = []
        self._slice = []
        gc.collect()
        
    def switch_to_ram(self, keep_data = True):
        """
        Switches data to a RAM based array.
        """
        
        if isinstance(self._data_pool, ram_data_pool):            
            # Already in RAM
            return 
            
        # Keep old dimension
        dim = self._data_pool._dim    

        new_pool = ram_data_pool(dim)
        
        if keep_data:            
            # First copy the data:
            new_pool.total = self._data_pool.total
                
        self._data_pool = new_pool          
        
        # Clean up!
        gc.collect()

        print('Switched data pool to RAM')
    
    def switch_to_swap(self, keep_data = False, swap_path = '/export/scratch3/kostenko/Fast_Data/swap', swap_name = 'swap'):
        """
        Switches data to an SSD based array.
        """
        #(swap_path, swap_name, dim, shape = None, dtype = 'float32'):

        # Swap path can be different
                        
        if keep_data:
            # First copy the data:
            shape = self._data_pool.shape
            dtype = self._data_pool.dtype   
            
            total = self._data_pool.total
            
            self._data_pool = swap_data_pool(swap_path, swap_name, shape, dtype)
            self._data_pool.total = total
                        
        else:
            # Create new:
            self._data_pool = swap_data_pool(swap_path, swap_name)
            
        # Clean up!
        gc.collect()
        
        print('Switched data pool to swap')        
              
    @property    
    def block_key(self):
        return self._block_key
        
    @block_key.setter
    def block_key(self, key):
        """
        Set a new buffer key.
        """
        if self._block_key != key:
            
            # Write buffer on disk if needed:
            self.finalize_slice()
            self.finalize_block()
            
            # Update indexing:
            self._block_key = key
            
    def change_block_size(self, block_sizeGB):
        """
        Change the block size and update the current block.
        """       
        self.block_key = -1 
        self._block = []

        self._block_sizeGB = block_sizeGB
        
        gc.collect()
                                
    def finalize_slice(self):
        """
        If buffer was modified - update file on disk.
        """
        if (self._slice_updated) & (self._slice_key >= 0):
            
            print(self._slice_updated)
            print(self._slice_key)
            print(self._slice.shape)
            
            self._data_pool.write(self._slice_key, self._slice)
            self._slice_updated = False

    def finalize_block(self):
        """
        If buffer (block) was modified - update data_pool.
        """        
        if self._block_updated & self._block_key >= 0:
            
            for ii, index in enumerate(self.block_index):
                self._data_pool.write(index, misc._get_dim_data(self._block, self.dim, ii))
                                
            self._block_updated = False
        
    def get_slice(self, key, dim = None):
        """
        Get one slice.
        """  
        
        if not dim is None:
            if dim != self.dim: raise Exception('data_blocks_swap can only retrieve slices along its main dimension.')
                                   
        # Do we have a buffer of that slice?
        if key == self._slice_key:
            return self._slice
        
        # Write swap on disk if it was updated...
        self.finalize_slice()
        
        # Check if we have the slice in the current block buffer:
        block_index = self.block_index        

        if key in block_index:
            # Use existing block:
            img = misc._get_dim_data(self._block, self.dim, numpy.where(block_index == key))
            
            print(block_index)
            print(key)
            print(numpy.where(block_index == key))
            
            #print('+++', img.shape)    
                
            return img
            
        else:
        
            # Upload data from disk if no buffer availbale:
            self._slice = self._data_pool.read(key)
            #print('***', self._slice.shape)
            # Update index key:
            self._slice_key = key     
            
            return self._slice
                            
    def set_slice(self, key, image):
        """
        Set one slice.
        """   
        if image.ndim > 3:
            print(image.shape)
            
        # Do we have a buffer of that slice?
        if (key != self._slice_key) | (key == (self.length - 1)):
            self.finalize_slice()
        
        # Write to buffer:    
        self._slice = image
        self._slice_updated = True
        
        # Are we in the right block? .... write to the block
        
        block_index = self.block_index
        
        if (key in block_index):
            # Use existing block:
            misc._set_dim_data(self._block, self.dim, numpy.where(block_index == key), image)
                           
        # All went OK - update current key:
        self._slice_key = key     

            
    def __iter__(self):
        """
        Return itarator for the array.
        """
        # Set block_key to -1 to make sure that the first iterator will return key = 0
        self.block_key = -1     
                
        return self  
        
    def __next__(self):
        """
        Retrun next block of data.
        """
        
        if self.block_key < (self.block_number - 1):
            
            return self[self.block_key + 1]

        else:     
            # End loop, update block:
            self.finalize_slice()
            self.finalize_block()            
            
            raise StopIteration
        
    def __getitem__(self, key):
        """
        Get block of data.
        """
        
        # If it is a current buffer:
        if self._block_key == key:    
            return self._block
            
        else:
            # Update buffer key:
            self.block_key = key
        
            # initialize buffer
            self._block = self.empty_block()

            # Read from swap:        
            for ii, index in enumerate(self.block_index):
                
                # Set data:
                misc._set_dim_data(self._block, self.dim, ii, self._data_pool.read(index))    
                        
            return self._block
        
    def __setitem__(self, key, data):    
        """
        Set block of data. Use it with care, it will write on SSD every time!
        """     
        
        # If the block moves to a new position:    
        self.block_key = key
                        
        # Update RAM buffer:
        self._block = data    
        self._block_updated = True
        
        # Update dtype:
        if self._my_dtype != data.dtype:
            if self._my_dtype is None: 
                self._my_dtype = data.dtype
            else:
                raise ValueError('Type of the data has changed from ' + str(self._my_dtype) + ' to ' + str(data.dtype))

    @property    
    def total(self):
        """
        Get all data.
        """        
        return self._data_pool.total
                
    @total.setter
    def total(self, array):
        """
        Set all data.
        """
        self._my_shape = array.shape
        self._my_dtype = array.dtype
        
        self._data_pool.total = array
        
        self._update_global_index()
        
    def init_total(self, shape):
        """
        Initialize an empty array of a given shape. Can now write individual blocks.
        """
        self._my_shape = shape        
        self._update_global_index()
        
    @property
    def shape(self):
        """
        Dimensions of the array.
        """
        return self._data_pool.shape
        
    @property
    def dim(self):
        """
        Main dimension of the data array
        """
        return self._data_pool._dim
        
    @dim.setter
    def dim(self, dim):
        """
        Main dimension of the data array
        """
        self._data_pool._dim = dim
        
    @property
    def dtype(self):
        """
        Data type of the array.
        """
        return numpy.dtype(self._data_pool.dtype)
            
    def _update_global_index(self):
        '''
        Create a global index for slices. Alowes to have everything in sequential, random or equidistant order.
        '''
        
        if self._indexer == 'sequential':
            # Index = 0, 1, 2, 4
            self._global_index = numpy.arange(self.length)
            
        elif self._indexer == 'random':   
            # Index = 2, 3, 0, 1 for instance...        
            self._global_index = random.shuffle(numpy.arange(self.length))    
            
        
        elif self._indexer == 'equidistant':   
            # Index = 0, 2, 1, 3    
            self._global_index = numpy.arange(self.length)
            
    @property    
    def block_index(self):
        """
        Get slice indexes for the current buffer key
        """
        
        if self._block_key < 0:
            return numpy.array([])

        block_step = int(numpy.ceil(self.length / self.block_number))
        
        start =  self._block_key * block_step
        stop = ( self._block_key + 1) * block_step
        
        # Check if we are still inside the array:
        if start >= self.length:
            raise IndexError('Block array is out of range.')
            
        stop = min((stop, self.length)) 
        
        return numpy.array(self._global_index[start:stop])
    
                        
    def empty_block(self, val = 0):
        """
        Make a block of zeroes.
        """
        shp = self.block_shape   
           
        if val == 0:
            return numpy.zeros(shp, dtype = numpy.float32)    
        else:
            return numpy.ones(shp, dtype = numpy.float32) * val    
        
    def empty_slice(self, val = 0):
        """
        Make a slice of zeroes.
        """
        sz = self.slice_shape
        
        if val == 0:
            return numpy.zeros(sz, dtype = numpy.float32)
        else:
            return numpy.ones(sz, dtype = numpy.float32) * val
        
    def __len__(self):
        """
        Number of blocks.
        """
        return self.block_number
    
    def block_xyz(self):
        """
        Return current block indexes. Can be used to create a meshgrid for instance.
        """
        
        if self.block_key == -1:
            raise ValueError('Block is undefined!')
            
        # Current block:
        shp = self.block_shape
        
        x = numpy.arange(0, shp[0])
        y = numpy.arange(0, shp[1])
        z = numpy.arange(0, shp[2])

        if self._data_pool._dim == 0:
            x = self.block_index
        elif self._data_pool._dim == 1:
            y = self.block_index
        else:
            z = self.block_index

        return [x,y,z]
            
    @property
    def slice_shape(self):
        """
        Dimensions of a single slice:
        """
        
        shp = self._data_pool.shape
        
        index = numpy.arange(3)
        index = index[index != self._data_pool._dim]
        
        return shp[index]
    
    @property    
    def block_shape(self):
        """
        Shape of the block. It might overestimate the shape of the last block, be careful!
        """
        sz = self._data_pool.shape
        
        if sz.size > 0:
            
            # Use actual index to take into account that the last block can be shorter 
            sz[self._data_pool._dim] = self.block_index.size

        else:
            raise ValueError('Data shape was not initialized yet!')
           
        return sz
            
    @property
    def length(self):
        """
        Length of the array along the blocking direction.
        """
        return self.shape[self.dim]  

    @property
    def block_number(self):
        """
        Number of blocks.
        """
        return int(1 + self.sizeGB / self._block_sizeGB)
        
    @property
    def size(self):
        """
        Number of elements of the array.
        """
        return self._data_pool.shape.prod()
                
    @property
    def sizeGB(self):
        """
        Size of the array in Gigabytes
        """
        return self.size / 1073741824 * self.dtype.itemsize    
      
# **************************************************************
#           IO class
# **************************************************************
class io(misc.subclass):
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
                a = numpy.zeros(data.shape[1:], dtype = numpy.float32)
                
            if a.ndim > 2:
              a = a.mean(2)
              
            data[ii, :, :] = a
    
            misc.progress_bar((ii+1) / numpy.shape(files)[0] * index_step)
            #print("\r Progress {:2.1%}".format(), end=" ")        
            
    
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
            path_folder = os.path.dirname(path_folder)

            # if file name is provided, the range is needed:
            if index_range != []:    
                first = index_range[0]
                last = index_range[1]

        else:
            # Try to find how many files do we need to read:

            # Get the files only:
            files = [x for x in os.listdir(path_folder) if os.path.isfile(os.path.join(path_folder, x))]

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
            filename = [x for x in os.listdir(path_folder) if str(last) in x][0]

            # Find the file with the minimum index:
            filename = sorted([x for x in os.listdir(path_folder) if (filename[:-8] in x)&(filename[-3:] in x)])[0]

            print('Reading a stack of images')
            print('Seed file name is:', filename)
            if index_step > 0:
                print('Reading every %d-nd(rd) image.' % index_step)
            #srt = self.settings['sort_by_date']AMT24-25-SU1/

            data = io._read_image_stack(os.path.join(path_folder,filename), index_step)
            
            if (index_range != []):
                data = data[index_range[0]:index_range[1], :, :]

            if (y_roi != []):
                data = data[:, y_roi[0]:y_roi[1], :]
            if (x_roi != []):
                data = data[:, :, x_roi[0]:x_roi[1]]

            return data
        
    @staticmethod    
    def _write_image(path_file, image):
        """
        File writer.
        """
        im = Image.fromarray(numpy.flipud(image.squeeze()))
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
           
    def read_projections(self, path, filter = ''):
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
        
    def read_slices(self, path, filter = ''):
        '''
        Read volume slices.
        '''        
        data = self._read_images(path, filter) 
        
        self._parent.data.total = data
        
        # Collect garbage:
        gc.collect()
        
        # add record to the history:
        self._parent.meta.history.add_record('io.read_slices', path)
        
    def write_slices(self, path, file_name = 'data'):
        '''
        Read volume slices.
        '''        
        
        data = self._read_images(path, filter) 
        
        self._parent.data.total = data
        
        # Collect garbage:
        gc.collect()
        
        # add record to the history:
        self._parent.meta.history.add_record('io.read_slices', path)    
        
    def _parse_unit(self, string):
        '''
        Look for units in the string and return a factor that converts this unit to Si.
        '''
        # Look at the inside a braket:
        if '[' in string:
            string = string[string.index('[')+1:string.index(']')]
                        
        elif '(' in string:
            string = string[string.index('(')+1:string.index(')')]

        else:
            return 1 
            
        # Here is what we are looking for:
        units_dictionary = {'um':0.001, 'mm':1, 'cm':10.0, 'm':1e3, 'rad':1, 'deg':numpy.pi / 180.0, 'ms':1, 's':1e3, 'us':0.001, 'kev':1, 'mev':1e3, 'ev':0.001,
                            'kv':1, 'mv':1e3, 'v':0.001, 'ua':1, 'ma':1e3, 'a':1e6, 'line':1}    
                            
        factor = [units_dictionary[key] for key in units_dictionary.keys() if key == string]
        
        if factor == []: factor = 1
        else: factor = factor[0]

        return factor             
        
    def _parse_keywords(self, path, keywords, file_mask = '.log', separator = ':'):
        '''
        Parse a text file using the keywords dictionary and create a dictionary with values.
        
        Args:
                path : Path to file
                file_mask : Part of the file's name
                separator : Seprates names of records from values in the log file
                keywords: Keywords to look for inside the file
                
         Returns:
               dictionary of the parsed values.
        '''
        #path = self._update_path(path)

        # Try to find the log file in the selected path
        log_file = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and file_mask in os.path.join(path, x))]

        # Check if there is one file:
        if len(log_file) == 0:
            raise FileNotFoundError('Log file not found in path: ' + path)
            
        if len(log_file) > 1:
            self._parent.warning('Found several log files. Currently using: ' + log_file[0])
            log_file = os.path.join(path, log_file[0])
        else:
            log_file = os.path.join(path, log_file[0])

        # Create an empty dictionary:
        records = {}

        # Loop to read the file record by record:
        with open(log_file, 'r') as logfile:
            for line in logfile:
                name, var = line.partition(separator)[::2]
                name = name.strip().lower()
                
                # If name contains one of the keys (names can contain other stuff like units):
                geom_key = [keywords[key] for key in keywords.keys() if key in name]

                # Collect record values:
                if geom_key != []:
                    
                    # Look for unit description in the name:
                    factor = self._parse_unit(name)
                    
                    # If needed separate the var and save the number of save the whole string:
                    try:
                        records[geom_key[0]] = float(var.split()[0]) * factor
                    except:
                        records[geom_key[0]] = var

        return records
       
    def _keywords_to_geometry(self, records):
        '''
        Convert parsed records to parameters in the geometry class.
        '''
        meta = self._parent.meta
                
        # Convert the geometry dictionary to geometry object:        
        meta.geometry.src2obj = records.get('src2obj')
        meta.geometry.det2obj = records.get('src2det') - records.get('src2obj')        
        meta.geometry.img_pixel = [records.get('img_pixel') * self._parse_unit('[um]'), records.get('img_pixel') * self._parse_unit('[um]')]  

        try:                                                
            meta.geometry.theta_range = [records.get('first_angle') * self._parse_unit('[deg]'), records.get('last_angle') * self._parse_unit('[deg]')]
        except:
            meta.geometry.theta_range = [0* self._parse_unit('[deg]'), 360* self._parse_unit('[deg]')]
        
        if self._parent.data != []:
            meta.geometry.init_thetas(theta_n = self._parent.data.shape[1])
            
        # Set some physics properties:
        meta.physics['binning'] = records.get('binning')
        meta.physics['averages'] = records.get('averages')
        meta.physics['voltage'] = records.get('voltage')
        meta.physics['power'] = records.get('power')
        meta.physics['current'] = records.get('current')
        meta.physics['exposure'] = records.get('exposure')
        meta.physics['mode'] = records.get('mode')
        meta.physics['filter'] = records.get('filter')
        
        #print(records.get('name'))
        meta.lyrics['sample name'] = records.get('name')
        meta.lyrics['comments'] = records.get('comments')
        meta.lyrics['date'] = records.get('date')
        meta.lyrics['duration'] = records.get('duration')
                
    def _keywords_to_flex(self, records):    
        
        meta = self._parent.meta
        
        # Fix some Flex ray hardware-specific stuff:
        records['det_hrz'] += 24
        
        records['src_vrt'] -= 5
        vol_center = (records['det_vrt'] + records['src_vrt']) / 2
        records['vol_z_tra'] = vol_center

        # Rotation axis:
        records['axs_hrz'] += 0.5
        
        # Compute the center of the detector:
        roi = numpy.int32(records.get('roi').split(sep=','))
        centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
        
        # Take into account the ROI of the detector:
        records['det_vrt'] += centre[1] / records.get('binning')
        records['det_hrz'] += centre[0] / records.get('binning')
            
        try:
            meta.geometry.det_trans[1] = records['det_vrt']
            meta.geometry.det_trans[0] = records['det_hrz']
            meta.geometry.src_trans[1] = records['src_vrt']
            meta.geometry.src_trans[0] = records['src_hrz']
            meta.geometry.axs_trans[0] = records['axs_hrz']
            meta.geometry.vol_trans[1] = records['vol_z_tra']

        except:
            print('Error parsing the motor positions!')
                              
    def parse_flexray(self, path):
        """
        Use this routine to parse 'scan settings.txt' file generated by FlexRay machine.
        """
        # Dictionary that describes the Flexray file:        
        keywords = {'voxel size':'img_pixel',
                        'sod':'src2obj',
                        'sdd':'src2det',
                        
                        'ver_tube':'src_vrt',
                        'ver_det':'det_vrt',
                        'tra_det':'det_hrz',
                        'tra_obj':'axs_hrz',
                        'tra_tube':'src_hrz',
                        
                        'Sample name' : 'comments',
                        'Comment' : 'name',
                        
                        '# projections':'theta_n',
                        'last angle':'last_angle',
                        'start angle':'first_angle',
                        'tube voltage':'voltage',
                        'tube power':'power',
                        'number of averages':'averages',
                        'imaging mode':'mode',
                        'binning value':'binning',
                        'roi (ltrb)':'roi',
                        'date':'date',
                        'scan duration':'duration',
                        'filter':'filter',
                        
                        'exposure time (ms)':'exposure'}

        # Read recods from the file:
        records = self._parse_keywords(path, keywords, file_mask = 'settings.txt', separator = ':')
        
        # Parse the file:
        self._keywords_to_geometry(records)
                
        # Translate the records to modifiers:
        self._keywords_to_flex(records)    
            
    def read_flexray(self, path):
        '''
        Read raw projecitions, dark and flat-field, scan parameters,
        '''
        self.read_dark(path + '/di0000.tif')
        
        self.read_ref([path + '/io0000.tif', path + '/io0001.tif'])
        
        self.read_projections(path)
        
        self.parse_flexray(path)
    