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

import misc

# **************************************************************
#           DATA_BLOCKS class
# **************************************************************
class data_blocks(object):
    """
    This will be a generic data array stored in RAM. Other types of arrays
    will be inherited from it (GPU, SSD etc.).
    
    User can get a block of data from the data_blocks by iterating it. 
    Or use 'total' property to set/get all data at once.
    """
    # Da data, in RAM version of data_blocks, _buffer contains the entire data:
    _buffer = []

    # Block index (in slice numbers):
    _index = []
   
    # Maximum block size in GB. 
    _block_sizeGB = 1
    
    # Dimension of access (dim = 1 for ASTRA projections, dim = 0 for volume slices) 
    _dim = 0
    
    # Random block iterator for stochastic methods
    random_iterator = False
    
    
    def __init__(self, array = [], block_sizeGB = 4, dim = 1):
                
        self.total = numpy.array(array, dtype = 'float32')
        
        self._dim = dim
        self._block_sizeGB = block_sizeGB
        
        print('Hello data_blocks!')
        
    def __del__(self): 
        
        self._buffer = []
        gc.collect()
        
        print('Bye bye data_blocks!')
                    
    @staticmethod   
    def _set_dim_data(data, dim, key, image):    
        """
        Sets a slice of data along a particular dimension:
        """
        if dim == 0:        
            data[key, :, :] = image

        elif dim == 1:
             data[:, key, :] = image

        else:
            data[:, :, key] = image
        
    @staticmethod    
    def _get_dim_data(data, dim, key):
        """
        Gets a slice of data along a particular dimension:
        """
        if dim == 0:        
            return data[key, :, :] 

        elif dim == 1:
            return data[:, key, :]

        else:
            return data[:, :, key]
    
    def __iter__(self):
        """
        Return itarator for the array.
        """
        # Reset the counter:
        self._buffer_key = -1
        self._index = []
                
        return self
                
    def _update_index(self, key):
        """
        Translate key into array index depending on the current block number.
        """
        block_step = int(numpy.ceil(self.length / self.block_number))
        
        start = key * block_step
        stop = (key + 1) * block_step
        
        # Check if we are still inside the array:
        if start >= self.length:
            raise IndexError('Block array is out of range.')
            
        stop = min((stop, self.length)) 
        
        self._index = numpy.arange(start, stop)
        
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

    def get_slice(self, key, dim = None):
        """
        Get one slice.
        """        
        if dim is None:
            dim = self._dim
        
        if dim == 0:        
            return self._buffer[key, :, :] 

        elif dim == 1:
             return self._buffer[:, key, :] 

        else:
            return self._buffer[:, :, key] 

    def set_slice(self, key, image):
        """
        Set one slice.
        """
       
        if self._dim == 0:        
            self._buffer[key, :, :] = image
        elif self._dim == 1:
             self._buffer[:, key, :] = image
        else:
            self._buffer[:, :, key] = image

    def __getitem__(self, key):
        """
        Get block of data.
        """
        self._update_index(key)    
        data = self._get_dim_data(self._buffer, self._dim, slice(self._index[0], self._index[-1] + 1))
        
        return numpy.ascontiguousarray(data)      
        
    def __setitem__(self, key, data):    
        """
        Set block of data.
        """
        self._update_index(key)   
        
        self._set_dim_data(self._buffer, self._dim, slice(self._index[0], self._index[-1] + 1), data)      
        
    def get_random_block(self):
        """
        Generate a block using random sampling
        """
        # Total index:
        index = numpy.arange(self.length)
        
        # Create block:
        block = self.empty_block()
        block_len = self.block_shape[self.dim]
        
        self._index = numpy.zeors(block_len)
        
        for ii in range(block_len):
            
            # Chose a random key:
            rand_index = numpy.random.randint(0, index.length)
            
            # Index of the slice to upload:
            key = index[rand_index]
            
            # Read image and add it to the block:
            self._set_dim_data(block, self._dim, ii, self.get_slice(key))  
            
            # This makes sure that indexes will not be sampled twice in one block
            index = numpy.delete(index, rand_index)
            
            # Add this key to my current indexes (used to generate thetas).
            self._index[ii] = key
            
        return numpy.ascontiguousarray(block)    
        
    def __len__(self):
        """
        Number of blocks.
        """
        return self.block_number

    def __next__(self):
        """
        Retrun next block of data.
        """
        if self.random_iterator:
            
            # Create block useing random sampling:
            return self.get_random_block()
        
        else:
            if self._buffer_key < (self.block_number - 1):
                
                self._buffer_key += 1
                block = self[self._buffer_key]
    
            else:     
                # End loop, update block:
                raise StopIteration
                        
            return block 
    
    def block_xyz(self):
        """
        Return current block indexes. Can be used to create a meshgrid for instance.
        """
        # Current block:
        key = self._buffer_key      
        shp = self.block_shape
        
        x = numpy.arange(0, shp[0])
        y = numpy.arange(0, shp[1])
        z = numpy.arange(0, shp[2])

        if self._dim == 0:
            x += self.index[0]
        elif self._dim == 1:
            y += self.index[0]
        else:
            z += self.index[0]        

        return [x,y,z]
    
    @property    
    def total(self):
        """
        Get all data.
        """
        return numpy.ascontiguousarray(self._buffer)
        
    @total.setter
    def total(self, array):
        """
        Set all data. 
        """ 
        self._buffer = array   
        
        
    @property
    def shape(self):
        """
        Dimensions of the array.
        """
        return numpy.array(self._buffer.shape) 
    
    @property
    def slice_shape(self):
        """
        Dimensions of a single slice:
        """
        
        shp = self.shape
        
        index = numpy.arange(3)
        index = index[index != self._dim]
        
        return shp[index]
    
    @property    
    def block_shape(self):
        """
        Shape of the block:
        """
        sz = self.shape
        
        if sz.size > 0:
            sz[self._dim] /= self.block_number

        else:
            raise ValueError('Data shape was not initialized yet!')
           
        return sz
           
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
        return int(1 + self.sizeGB / self._block_sizeGB)
        
    @property
    def size(self):
        """
        Number of elements of the array.
        """
        return self.shape.prod()
        
    @property
    def dtype(self):
        """
        Data type of the array.
        """
        return self._buffer.dtype
        
    @property
    def sizeGB(self):
        """
        Size of the array in Gigabytes
        """
        return self.size / 1073741824 * self.dtype.itemsize    
 
# **************************************************************
#           data_blocks_ssd class
# **************************************************************
class data_blocks_ssd(data_blocks):
    """
    This class doesn't keep data in RAM memory. It has a swap storage on SSD disk 
    and keeps only one block of data in the RAM buffer.
    
    At the moment there are two types of access: 
        1. read/write access to block
        2. read/write access to a slice
    """
    # Block buffer and it's current key:
    _buffer = []
    _buffer_key = None
    _buffer_updated = False
    
    # Remeber the shape of the total data on disk:
    _my_shape = []
    _my_dtype = None
    
    # Additional buffer for the slice image:
    _buffer_slice = []
    _buffer_slice_key = None
    _buffer_slice_updated = False
    
    # SSD swap:
    _swap_path = ''
    _swap_name = 'swap'
    
    def __init__(self, array = None, shape = None, dtype = None, block_sizeGB = 1, dim = 1, swap_path = ''):
        
        self._dim = dim
        self._block_sizeGB = block_sizeGB
        
        self._swap_path = swap_path
        if not os.path.exists(self._swap_path):
            os.makedirs(self._swap_path)
        
        if array != None:
            self.total = numpy.array(array, dtype = 'float32')
            
        elif (shape != None) & (dtype != None):
            self._my_dtype = numpy.dtype(dtype)
            self._my_shape = shape 
        
    def __del__(self):
        
        # Double check that the last buffer was written on disk:
        self.finalize_buffer_slice()  
        self.finalize_buffer()  
        
        # Free memory:
        self._buffer = []
        gc.collect()
        
        # Free SSD:
        self._remove_swap()
        
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
                                
    def finalize_buffer_slice(self):
        """
        If buffer was modified - update file on disk.
        """
        if self._buffer_slice_updated:
            
            self._write_swap(self._buffer_slice_key, self._buffer_slice)
            self._buffer_slice_updated = False
        
    def finalize_buffer(self):
        """
        If buffer (block) was modified - update files on disk.
        
        Attr:
            key: move to htat key after writing last one on disk.
        """        
        if self._buffer_updated:
            
            # Check if swappin path exists:
            if not os.path.exists(self._swap_path):
                os.makedirs(self._swap_path)
                
            start, stop = self._get_index(self._buffer_key)
            
            #print('writing data from:', start, stop )
            
            for ii in range(start, stop):
                self._write_swap(ii, self._get_dim_data(self._buffer, self._dim, [ii - start]))
                
            self._buffer_updated = False
        
    def get_slice(self, key, dim = None):
        """
        Get one slice.
        """  
        
        if not dim is None:
            if dim != self._dim: raise Exception('data_blocks_ssd can only retrieve slices along its main dimension.')
                                   
        # Do we have a buffer of that slice?
        if key == self._buffer_slice_key:
            return self._buffer_slice
        
        # Write swap on disk if it was updated...
        self.finalize_buffer_slice()
        
        # Check if we have the slice in the current block buffer:        
        start, stop = self._get_index(self._buffer_key)
        
        if (key >= start) & (key < stop):
            # Use existing block:
            return self._get_dim_data(self._buffer, self._dim, key - start)
        
        # Upload data from disk if no buffer availbale:
        self._buffer_slice = self._read_swap(key)
        
        # Update index key:
        self._buffer_slice_key = key     
        
        return self._buffer_slice
                            
    def set_slice(self, key, image):
        """
        Set one slice.
        """        
        # Do we have a buffer of that slice?
        if (key != self._buffer_slice_key) | (key == (self.length - 1)):
            self.finalize_buffer_slice()
        
        # Write to buffer:    
        self._buffer_slice = image
        self._buffer_slice_updated = True
        
        # Are we in the right block? .... write to the block
        start, stop = self._get_index(self._buffer_key)
                
        if (key >= start) & (key < stop):            
            # Use existing block:
            self._set_dim_data(self._buffer, self._dim, key - start, image)
           
        # Update dtype:
        if self._my_dtype != image.dtype:
            if self._my_dtype is None: 
                self._my_dtype = image.dtype
            else:
                raise ValueError('Type of the data has changed from ' + str(self._my_dtype) + ' to ' + str(image.dtype))
                
        # All went OK - update current key:
        self._buffer_slice_key = key     

            
    def __iter__(self):
        """
        Return itarator for the array.
        """
        
        # Reset the buffers:
        self.finalize_buffer_slice()
        self.finalize_buffer()
        
        # Set buffer_key to -1 to make sure that the first iterator will return key = 0
        self._buffer_key = -1     
                
        return self  
        
    def __next__(self):
        """
        Retrun next block of data.
        """
        
        if self.random_iterator:
            
            # Create block useing random sampling:
            return self.get_random_block()
        
        else:
            # If random iterator is off, use the buffer:
            if self._buffer_key < (self.block_number - 1):
                
                self._buffer_key += 1
                block = self[self._buffer_key]
    
            else:     
                # End loop, update block:
                self.finalize_buffer_slice()
                self.finalize_buffer()            
                
                raise StopIteration
            
            #print('returning block ',    self._buffer_key)
            return block     
        
    def __getitem__(self, key):
        """
        Get block of data.
        """
        
        # If it is a current buffer:
        if self._buffer_key == key:    
            print('get item(old)', self._buffer_key, self._buffer.shape)   
            return self._buffer
            
        # Write swap on disk if it was updated...
        self.finalize_buffer_slice()
        self.finalize_buffer()
        
        # Read new swap:
        self._update_index(key)
        
        # initialize buffer
        self._buffer = self.empty_block()

        # Read from swap:        
        for ii in self._index:
            # Set data:
            self._set_dim_data(self._buffer, self._dim, key, self._read_swap(ii))    
                        
        self._buffer_key = key
        
        print('get item(new)', self._buffer_key, self._buffer.shape)   
        
             
        return self._buffer
        
    def __setitem__(self, key, data):    
        """
        Set block of data. Use it with care, it will write on SSD every time!
        """     
        
        # If the block moves to a new position:    
        if self._buffer_key != key:            
                    
            # Update slice buffer:
            self.finalize_buffer_slice()
            
            # Update block buffer:  
            self.finalize_buffer()    
                        
        # Update RAM buffer:
        self._buffer = data    
        self._buffer_updated = True
        self._buffer_key = key     
        #print('set item', self._buffer_key)         
        
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
        
        array = numpy.zeros(self.shape, dtype = self._my_dtype)
        
        for jj in range(self.block_number):
            
            # block index limits:
            self._update_index(jj)
            
            # write block:
            self._set_dim_data(array, self._dim, slice(self.index[0], self.index[-1] + 1), self[jj])      
        
        return array 
        
    @total.setter
    def total(self, array):
        """
        Set all data.
        """
        
        self._my_shape = array.shape
        self._my_dtype = array.dtype
        
        for jj in range(self.block_number):
            
            # block index limits:
            self._update_index(jj)
            
            # write block:
            self[jj] = self._get_dim_data(array, self._dim, slice(self.index[0], self.index[-1] + 1))          
    
    def init_shape(self, shape):
        """
        Initialize an empty SSD array of a given shape. Can now write individual blocks.
        """
        self._my_shape = shape
        
    @property
    def shape(self):
        """
        Dimensions of the array.
        """
        return numpy.array(self._my_shape)
        
    @property
    def dtype(self):
        """
        Data type of the array.
        """
        return self._my_dtype    

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
        
        # Compute the center of the detector:
        roi = numpy.int32(records.get('roi').split(sep=','))
        centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
        
        # Take into account the ROI of the detector:
        records['det_vrt'] += centre[1] / records.get('binning')
        records['det_hrz'] += centre[0] / records.get('binning')
            
        meta.geometry.det_trans[1] = records['det_vrt']
        meta.geometry.det_trans[0] = records['det_hrz']
        meta.geometry.src_trans[1] = records['src_vrt']
        meta.geometry.src_trans[0] = records['src_hrz']
        meta.geometry.axs_trans[0] = records['axs_hrz']
        meta.geometry.vol_trans[2] = records['vol_z_tra']
                              
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
    