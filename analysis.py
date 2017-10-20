#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains routines related to the data analysis and processing for the tomobox.
"""

import numpy
from scipy import ndimage           # Image processing, filters
import gc

import matplotlib.pyplot as plt

# Tomobox modules
import misc 
import simulate        
# **************************************************************
#           ANALYSE class
# **************************************************************        
class analyse(misc.subclass):
    '''
    This is an anlysis toolbox for the raw and reconstructed data
    '''
    
    def l2_norm(self):
        norm = 0
        
        for block in self._parent.data:
            norm += numpy.sum(block ** 2)
            
        return norm
       
    def sum(self, dim = None):      
        
        # If no dimension provided sum everything:
        if dim is None:
            vals = 0

            for block in self._parent.data:
                vals += numpy.sum(block)
            
            return vals
            
        # if dim is the same as the main dimension 
        if dim == self._parent.data.dim:
            
            img = self._parent.data.empty_slice()   
            
            for block in self._parent.data:
                img += numpy.sum(block, dim)
                
            return img
            
        # if dim is not the same as the main dimension     
        else:
            print('Warning! Summing perpendicular to the main direction of the data array is not implemented for SSD arrays!')
            
            return numpy.sum(self._parent.data.total, dim)
            
    def mean(self, dim = None):
        """
        Compute mean.
        """
        return self.sum(dim) / self._parent.data.size

    def min(self, dim = None):
        """
        Compute minimum.
        """
        
        # Initial:
        if dim is not None:
            val = self._parent.data.empty_slice(numpy.inf)
        else:
            val = numpy.inf
       
        # Min    
        for block in self._parent.data:
            if dim is not None:
                val = numpy.min((val, numpy.min(block, dim)), 0)
            else:
                val = numpy.min((val, numpy.min(block)))
            
        return val

    def max(self, dim = None):
        """
        Compute maximum.
        """
        # Initial:
        if dim is not None:
            val = self._parent.data.empty_slice(-numpy.inf)
        else:
            val = -numpy.inf
                
        for block in self._parent.data:
            if dim is not None:
                val = numpy.max((val, numpy.max(block, dim)), 0)
                
            else:
                val = numpy.max((val, numpy.max(block)))
            
        return val
        
    def percentile(self, prcnt):
        """
        Compute percentile.
        """
        val = 0
        for block in self._parent.data:
            val += numpy.percentile(block, prcnt)
            
        return val / len(self._parent.data)    

    def center_of_mass(self, dim = 1):
        '''
        Return the center of mass**2 (power is there to avoid influence of small values).
            Arg:
                dim (int): dimension that is averaged before the center of mass is computed.
        '''
        
        data = self._parent.data
        
        # Indexes and masses:
        xx, yy, zz = 0, 0, 0
        m = 0
        
        for block in data:
            xyz = data.block_xyz()
            
            m2 = block ** 2
            xx += numpy.sum(xyz[0] * m2)
            yy += numpy.sum(xyz[1] * m2)
            zz += numpy.sum(xyz[2] * m2)
            
            m += numpy.sum(m2)
            
        return xx / m, yy / m, zz / m    

    def histogram(self, nbin = 256, plot = True, log = False, slice_num = []):

        mi = self.min()
        ma = self.max()

        vals = numpy.zeros(nbin)
        
        if slice_num == []:
            for block in self._parent.data:
                a, b = numpy.histogram(block, bins = nbin, range = [mi, ma])
                vals += a

        else:
            vals, b = numpy.histogram(self._parent.data.get_slice(slice_num), bins = nbin, range = [mi, ma])      

        # Set bin values to the middle of the bin:
        b = (b[0:-1] + b[1:]) / 2

        if plot:
            plt.figure()
            if log:
                plt.semilogy(b, vals)
            else:
                plt.plot(b, vals)
            plt.show()

        return vals, b
        
# **************************************************************
#           DISPLAY class and subclasses
# **************************************************************
class display(misc.subclass):
    """
    This is a collection of display tools for the raw and reconstructed data
    """
    
    def __init__(self, parent = []):
        misc.subclass.__init__(self, parent)
        
        self._cmap = 'magma'
        self._dynamic_range = []
        self._mirror = False
        self._upsidedown = False
        
    def set_options(self, cmap = 'magma', dynamic_range = [], mirror = False, upsidedown = False):    
        '''
        Set options for visualization.
        '''
        self._cmap = cmap
        self._dynamic_range = dynamic_range
        self._mirror = mirror
        self._upsidedown = upsidedown

    def _figure_maker_(self, fig_num):
        '''
        Make a new figure or use old one.
        '''
        if fig_num:
            plt.figure(fig_num)
        else:
            plt.figure()


    def slice(self, index = None, dim = 1, fig_num = []):
        '''
        Display a 2D slice of 3D volumel
        '''
        self._figure_maker_(fig_num)

        if index is None:
            index = self._parent.data.shape[dim] // 2

        img = self._parent.data.get_slice(index, dim)

        if self._mirror: img = numpy.fliplr(img)
        if self._upsidedown: img = numpy.flipud(img)
        
        if self._dynamic_range != []:
            plt.imshow(img, cmap = self._cmap, origin='lower', vmin = self._dynamic_range[0], vmax =self._dynamic_range[1])
        else:
            plt.imshow(img, cmap = self._cmap, origin='lower')
            
        # Avoid creating colorbar twice:        
        if fig_num == []:
            plt.colorbar()            

        plt.show()
        plt.pause(0.0001)

    def slice_movie(self, dim = 1, fig_num = []):
        '''
        Display a 2D slice of 3D volumel
        '''
        self._figure_maker_(fig_num)

        slice_num = 0
        img = self._parent.data.get_slice(slice_num, dim)
        fig = plt.imshow(img, cmap = self._cmap)

        plt.colorbar()
        plt.show()

        for slice_num in range(1, self._parent.data.length):
            img = self._parent.data(slice_num)
            fig.set_data(img)            
            plt.show()
            plt.title(slice_num)
            plt.pause(0.0001)

    def projection(self, dim = 1, fig_num = []):
        '''
        Get a projection image of the 3d data.
        '''
        self._figure_maker_(fig_num)

        # Dont know why, but the sum turns out up side down
        img = numpy.flipud(self._parent.analyse.sum(dim))
        
        if self._dynamic_range != []:
            plt.imshow(img, cmap = self._cmap, origin='lower', vmin = self._dynamic_range[0], vmax =self._dynamic_range[1])
        else:
            plt.imshow(img, cmap = self._cmap)
        
        plt.colorbar()
        plt.show()
        plt.pause(0.0001)
            
    def max_projection(self, dim = 1, fig_num = []):
        '''
        Get maximum projection image of the 3d data.
        '''
        self._figure_maker_(fig_num)

        img = self._parent.analyse.max(dim)
        
        if self._dynamic_range != []:
            plt.imshow(img, cmap = self._cmap, origin='lower', vmin = self._dynamic_range[0], vmax =self._dynamic_range[1])
        else:
            plt.imshow(img, cmap = self._cmap)
        
        plt.colorbar()
        plt.show()
        plt.pause(0.0001)

    def min_projection(self, dim = 1, fig_num = []):
        '''
        Get maximum projection image of the 3d data.
        '''
        self._figure_maker_(fig_num)

        img = self._parent.analyse.min(dim)
        
        if self._dynamic_range != []:
            plt.imshow(img, cmap = self._cmap, origin='lower', vmin = self._dynamic_range[0], vmax =self._dynamic_range[1])
        else:
            plt.imshow(img, cmap = self._cmap)
        
        plt.colorbar()
        plt.show()
        plt.pause(0.0001)

# **************************************************************
#           POSTPROCESS class
# **************************************************************       
class postprocess(misc.subclass):
    '''
    Processing for the volume data. Similar to process class, but takes
    into account volume specifics.
    '''
    
    def arbitrary_function(self, func):
        '''
        Apply an arbitrary function:
        '''
        process.arbitrary_function(self, func)
        
    def threshold(self, threshold = None, binary = True, morethan = True):
        '''
        Apply simple segmentation of value bounds.
        '''
        process.threshold(self, threshold = None, binary = True, morethan = True)
        
# **************************************************************
#           PROCESS class
# **************************************************************       
class process(misc.subclass):
    '''
    Various preprocessing routines
    '''
    
    def arbitrary_function(self, func):
        '''
        Apply an arbitrary function:
        '''
        print('Applying: ', func)
        
        for jj, block in enumerate(self._parent.data):
            
            # We need an index to update the content of an iterator:
            self._parent.data[jj] = func(block)
            
        # add a record to the history:
        self._parent.meta.history.add_record('process.arbitrary_function', func.__name__)

        self._parent.message('Arbitrary function applied.')

    def threshold(self, threshold = None, binary = True, morethan = True):
        '''
        Apply simple segmentation or discard small values.
        '''               
        if threshold is None:
            print('Computing half of the 99% percentile to use it as a threshold...')
            
            # Compute 50% intensity of the 99% of values (filter out outlayers).
            threshold = self._parent.analyse.percentile(99) / 2

            print('Applying threshold...')
            
            for jj, block in enumerate(self._parent.data):        
                if binary:
                    if morethan:
                        block = numpy.array(block > threshold, dtype = 'float32')
                    else:
                        block = numpy.array(block < threshold, dtype = 'float32')

                else:
                    
                    # Setting velues below threshold to zero:
                    if morethan:
                        block = block[block < threshold] = 0
                    else:
                        block = block[block > threshold] = 0

                self._parent.data[jj] = block
                misc.progress_bar(jj / self._parent.data.block_number)

        # add a record to the history:
        self._parent.meta.history.add_record('process.threshold', [threshold, binary, morethan])
        
    def interpolate_holes(self, mask2d, kernel = [3,3,3]):
        '''
        Fill in the holes, for instance, saturated pixels.
        
        Args:
            mask2d: holes are zeros. Mask is the same for all projections.
        '''
        
        for ii, block in enumerate(self._parent.data):    
                    
            # Compute the filler:
            tmp = ndimage.filters.gaussian_filter(mask2d, sigma = kernel)        
            tmp[tmp > 0] **= -1

            # Apply filler:                 
            block = block * mask2d[:, None, :]           
            block += ndimage.filters.gaussian_filter(block, sigma = kernel) * (~mask2d[:, None, :])
            
            self._parent.data[ii] = block   

            # Show progress:
            misc.progress_bar((ii+1) / self._parent.data.block_number)
            
        self._parent.meta.history.add_record('process.interpolate_holes(mask2d, kernel)', kernel)

    def residual_rings(self, kernel=[3, 1, 3]):
        '''
        Apply correction by computing outlayers .
        '''
        
        prnt = self._parent
        
        # Compute mean image of intensity variations that are < 5x5 pixels
        print('Our best agents are working on the case of the Residual Rings. This can take years if the kernel size is too big!')

        tmp = prnt.data.empty_slice()
        
        for ii, block in enumerate(self._parent.data):                 
            
            # Compute:
            tmp += (block - ndimage.filters.median_filter(block, size = kernel)).sum(1)
            
            misc.progress_bar((ii+1) / self._parent.data.block_number)
            
        tmp /= prnt.data.length
        
        print('Subtract residual rings.')
        
        for ii, block in enumerate(self._parent.data):
            block -= tmp[:, None, :]

            misc.progress_bar((ii+1) / self._parent.data.block_number)
            
            self._parent.data[ii] = block 

        prnt.meta.history.add_record('process.residual_rings(kernel)', kernel)
        
        print('Residual ring correcion applied.')

    def subtract_air(self, air_val = None):
        '''
        Subtracts a coeffificient from each projection, that equals to the intensity of air.
        We are assuming that air will produce highest peak on the histogram.
        '''
        prnt = self._parent
        print('Air intensity will be derived from 10 pixel wide border.')
        
        for ii, block in enumerate(self._parent.data):

            if air_val is None:  
                # Take pixels that belong to the 5 pixel-wide margin.
                border = numpy.array([block[:10, :], block[-10:, :], block[:, -10:], block[:, :10]])
              
                y, x = numpy.histogram(border, 2**12)
                x = (x[0:-1] + x[1:]) / 2
        
                # Subtract maximum argument:    
                val = x[y.argmax()]
                        
                block = block - val
            else:
                block = block - air_val
                
            self._parent.data[ii] = block

            misc.progress_bar((ii+1) / self._parent.data.block_number)
                    
        prnt.meta.history.add_record('subtract_air(air_val)', air_val)        
    
    def medipix_quadrant_shift(self):
        '''
        Expand the middle line
        '''
        
        print('Applying medipix pixel shift.')
        
        # this one has to be applied to the whole dataset as it changes its size
        data = self._parent.data.total
        
        misc.progress_bar(0)
        data[:,:, 0:data.shape[2]//2 - 2] = data[:,:, 2:data.shape[2]/2]
        data[:,:, data.shape[2]//2 + 2:] = data[:,:, data.shape[2]//2:-2]

        misc.progress_bar(0.5)

        # Fill in two extra pixels:
        for ii in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-ii) < numpy.abs(2-ii)) else 2
            data[:,:, data.shape[2]//2 - ii] = data[:,:, data.shape[2]//2 + closest_offset]

        misc.progress_bar(0.7)

        # Then in columns
        data[0:data.shape[0]//2 - 2,:,:] = data[2:data.shape[0]//2,:,:]
        data[data.shape[0]//2 + 2:, :, :] = data[data.shape[0]//2:-2,:,:]

        # Fill in two extra pixels:
        for jj in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-jj) < numpy.abs(2-jj)) else 2
            data[data.shape[0]//2 - jj,:,:] = data[data.shape[0]//2 + closest_offset,:,:]

        misc.progress_bar(0.8)

        self._parent.data.total = data
        
        misc.progress_bar(1)

        self._parent.meta.history.add_record('Quadrant shift', 1)
        print('Medipix quadrant shift applied.')

    def skyscan_flat(self):    
        '''
        Apply flat field like they do it in skyscan.
        '''
        prnt = self._parent
        
        self._parent.message('Applying flat field of a skyscan type...')
        
        if self._parent.meta.geometry.roi_fov:
            print('Object is larger than the FOV!')   
            
        for ii, block in enumerate(prnt.data):
            
            if self._parent.meta.geometry.roi_fov:
                air_values = numpy.ones(block.shape[:2]) * 2**16 - 1
            else:
                air_values = numpy.max(block, axis = 2)
            
            air_values = air_values.reshape((air_values.shape[0], air_values.shape[1],1))
    
            block = block / air_values
            
            self._parent.data[ii] = block
        
        # add a record to the history:
        self._parent.meta.history.add_record('process.skyscan_flat', 1)

        print('Skyscan flat field correction applied.')

    def flat_field(self):
        '''
        Apply flat field correction based on reference images.
        '''
        prnt = self._parent
        
        self._parent.message('Applying flat field....')
        
        # Go slice by slice:
        for ii in range(prnt.data.length):     
            
            img = prnt.data.get_slice(ii)
            
            #print('get slice', img.shape)
                        
            ref = self._parent.get_ref(proj_num = ii)
                
            if numpy.min(ref) <= 0:
                self._parent.warning('Flat field reference image contains zero (or negative) values! Will replace those with little tiny numbers.')

                tiny = ref[ref > 0].min()
                ref[ref <= 0] = tiny

            # If there is a dark field, use it.
            if not prnt._dark is None:    
                img  = img - prnt._dark

                # Is dark subtracted from ref images internally?
                ref = ref - prnt._dark
                
            # Use flat field:                
            #print('set slice', img.shape)    
            prnt.data.set_slice(ii, img / ref)

            misc.progress_bar((ii+1) / prnt.data.length)
            
        # add a record to the history:
        self._parent.meta.history.add_record('process.flat_field', 1)

        self._parent.message('Flat field correction applied.')

    def log(self, air_intensity = 1.0, lower_bound = -numpy.log(2), upper_bound = numpy.log(2**12)):
        '''
        Apply -log(x) to the sinogram. Lower and upper bounds are given for the attenuation coefficient.
        Default upper_bound assumes that values below 1/2^12 are outside of the dynamic range of the camera.
        Lover bound of - log(2) means there should be no intensity values higher than 2 after normalization.
        '''
        
        prnt = self._parent
        
        for ii, block in enumerate(prnt.data):
            
            if (air_intensity != 1.0):
                block /= air_intensity
            
            # Apply a bound to large values:
            numpy.clip(block, a_min = numpy.exp(-upper_bound), a_max = numpy.exp(-lower_bound), out = block)
                              
            # negative logarithm
            numpy.log(block, out = block)
            numpy.negative(block, out = block)
            
            # Save block
            prnt.data[ii] = block

            #print(ii)
            
            # Progress:
            misc.progress_bar((ii+1) / self._parent.data.block_number)
                    
        self._parent.message('Logarithm is applied.')
        self._parent.meta.history.add_record('process.log(air_intensity, bounds)', [air_intensity, lower_bound, upper_bound])

    def salt_pepper(self, kernel = [3, 1, 3]):
        '''
        Get rid of nasty speakles
        '''
        
        prnt = self._parent
        
        for ii, block in enumerate(prnt.data):
            # Make a smooth version of the data and look for outlayers:
            smooth = ndimage.filters.median_filter(block, kernel)
            mask = block / smooth
            mask = (numpy.abs(mask) > 2) | (numpy.abs(mask) < 0.5)

            block[mask] = smooth[mask]
            prnt.data[ii] = block

            misc.progress_bar((ii+1) / self._parent.data.block_number)

        self._parent.message('Salt and pepper filter is applied.')

        self._parent.meta.history.add_record('process.salt_pepper(kernel)', kernel)

    def add_noise(self, std, mode = 'normal'):
        '''
        Noisify the projection data!
        '''
        
        for ii, block in enumerate(self._parent.data):
            if mode == 'poisson':
                block[block < 0] = 0
                self._parent.data[ii] = numpy.random.poisson(block)
                
            elif mode == 'normal':
                self._parent.data[ii] = numpy.random.normal(block, std)
                
            else: 
                raise ValueError('Me not recognize the mode! Use normal or poisson!')
                
            misc.progress_bar((ii+1) / self._parent.data.block_number)
        
        self._parent.meta.history.add_record('process.add_noise(std, mode)', [std, mode])
        
    def simple_tilt(self, tilt):
        '''
        Tilts the sinogram
        '''
        
        prnt = self._parent
        
        print('Applying tilt.')
        for ii in range(prnt.data.length):     
            img = prnt.data.get_slice(ii)
            img = ndimage.interpolation.rotate(img, -tilt, reshape=False)
            
            prnt.data.set_slice(ii, img)
            
            misc.progress_bar(ii / prnt.data.length)
            
        prnt.message('Tilt is applied.')                 
        
    def crop(self, top_left, bottom_right):
        '''
        Crop the sinogram
        '''
        
        print('Cropping...')
        
        # Make sure there are no negative indexes:
        if top_left[0] < 0: top_left[0] = 0
        if bottom_right[0] < 0: bottom_right[1] = 0
        if top_left[1] < 0: top_left[0] = 0
        if bottom_right[1] < 0: bottom_right[1] = 0

        misc.progress_bar(0)
        
        # Since data size is changing we are going to apply crop to the total data:
        data = self._parent.data.total
            
        if bottom_right[1] > 0:
            data = data[top_left[1]:-bottom_right[1], :, :]
        else:
            data = data[top_left[1]:, :, :]

        misc.progress_bar(1/3)

        if bottom_right[0] > 0:
            data = data[:, :, top_left[0]:-bottom_right[0]]
        else:
            data = data[:, :, top_left[0]:]

        misc.progress_bar(2/3)
        
        # Put the data back:
        self._parent.data.total = data 
        
        misc.progress_bar(1)

        # Clean up
        gc.collect()
        
        if numpy.min(data.shape) == 0:
            self._parent.warning('Wow! We have just cropped the data to death! No data left.')

        self._parent.meta.history.add_record('process.crop(top_left, bottom_right)', [top_left, bottom_right])

        self._parent.message('Sinogram cropped.')
        
    def equivalent_thickness(self, energy, spectrum, compound, density):
        '''
        Transfer intensity values to equivalent thicknessb
        '''
        prnt = self._parent
        
        # Assuming that we have log data!
        if not 'process.log(air_intensity, bounds)' in self._parent.meta.history.keys:                        
            self._parent.error('Logarithm was not found in history of the projection stack. Apply log first!')
        
        print('Generating the transfer function.')
        
        # Attenuation of 1 mm:
        mu = simulate.spectra.linear_attenuation(energy, compound, density, thickness = 0.1)
        width = self._parent.data.slice_shape[1]

        # Make thickness range that is sufficient for interpolation:
        thickness_min = 0
        thickness_max = width * self._parent.meta.geometry.img_pixel[1]
        
        print('Assuming thickness range:', [thickness_min, thickness_max])
        thickness = numpy.linspace(thickness_min, thickness_max, 1000)
        
        exp_matrix = numpy.exp(-numpy.outer(thickness, mu))
        synth_counts = exp_matrix.dot(spectrum)
        
        plt.figure()
        plt.plot(thickness, synth_counts, 'r--', lw=4, alpha=.8)
        plt.axis('tight')
        plt.title('Intensity v.s. absorption length.')
        plt.show()
        
        synth_counts = -numpy.log(synth_counts)
        
        print('Callibration intensity range:', [synth_counts[0], synth_counts[-1]])
        print('Data intensity range:', [self._parent.analyse.min(), self._parent.analyse.max()])

        print('Applying transfer function.')    
        
        for ii, block in enumerate(self._parent.data):        
            block = numpy.array(numpy.interp(block, synth_counts, thickness), dtype = 'float32')
            
            prnt.data[ii] = block
            
            misc.progress_bar((ii+1) / self._parent.data.block_number)
            
        self._parent.meta.history.add_record('process.equivalent_thickness(energy, spectrum, compound, density)', [energy, spectrum, compound, density])    
                                              
    def bin_projections(self):
        '''
        Bin data with a factor of two in detector plane.
        '''
        print('Binning projection data.')    
        
        misc.progress_bar(0)
        
        data = self._parent.data.total
        data[:, :, 0:-1:2] += data[:, :, 1::2]
        data = data[:, :, 0:-1:2] / 2

        misc.progress_bar(0.5)

        data[0:-1:2, :, :] += data[1::2, :, :]
        data = data[0:-1:2, :, :] / 2

        misc.progress_bar(0.75)

        self._parent.data.total = data
        
        misc.progress_bar(1)

        # Multiply the detecotr pixel width and height:
        self._parent.meta.geometry.det_pixel[0] = self._parent.meta.geometry.det_pixel[0] * 2
        self._parent.meta.geometry.det_pixel[1] = self._parent.meta.geometry.det_pixel[1] * 2

        print('Binned.')
        self._parent.meta.history.add_record('process.bin_projections()', 2)
