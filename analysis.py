#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Oct 2017
@author: kostenko

This module contains routines related to the data analysis and processing for the tomobox.
"""

import numpy
from scipy.ndimage import measurements
import transforms3d
from transforms3d import euler
from skimage import morphology

import matplotlib.pyplot as plt

from misc import *
        
# **************************************************************
#           ANALYSE class
# **************************************************************        
class analyse(subclass):
    '''
    This is an anlysis toolbox for the raw and reconstructed data
    '''
    
    def l2_norm(self):
        norm = 0
        
        for block in self._parent._data:
            norm += numpy.sum(block ** 2)
            
        return norm
        
    def sum(self, dim = None):      
        block = self._parent._data[0]
        
        vals = numpy.sum(block, dim)

        for block in self._parent._data:
            vals += numpy.sum(block, dim)
            
        return vals

    def mean(self, dim = None):
        """
        Compute mean.
        """
        return self.sum(dim) / self._parent._data.size

    def min(self, dim = None):
        """
        Compute minimum.
        """
        val = 10**10
        
        block = self._parent._data[0]

        val = numpy.min(block, dim)
        
        for block in self._parent._data[1:]:
            val = numpy.min((val, numpy.min(block, dim)), 0)
            
        return val

    def max(self, dim = None):
        """
        Compute maximum.
        """
        val = -10**10
        
        block = self._parent._data[0]

        val = numpy.max(block, dim)
        
        for block in self._parent._data[1:]:
            val = numpy.max((val, numpy.max(block, dim)), 0)
            
        return val
        
    def percentile(self, prcnt):
        """
        Compute percentile.
        """
        val = 0
        for block in self._parent._data
            val += numpy.percentile(block, prcnt)
            
        return val / len(self._parent._data)    

    def center_of_mass(self, dim = 1):
        '''
        Return the center of mass**2 (power is there to avoid influence of small values).
            Arg:
                dim (int): dimension that is averaged before the center of mass is computed.
        '''
        
        data = self._parent._data
        sz = data.shape
        
        # Indexes and masses:
        xx, yy, zz = 0, 0, 0
        m = 0
        
        for block in data:
            x,y,z = data.block_index()
            
            m2 = block ** 2
            xx += numpy.sum(x * m2)
            yy += numpy.sum(y * m2)
            zz += numpy.sum(z * m2)
            
            m += numpy.sum(m2)
            
        return xx / m, yy / m, zz / m    

    def histogram(self, nbin = 256, plot = True, log = False, slice_num = []):

        mi = self.min()
        ma = self.max()

        vals = numpy.zeros(nbin)
        
        if slice_num == []:
            for block in self._parent._data:
                a, b = numpy.histogram(block, bins = nbin, range = [mi, ma])
                vals += a

        else:
            vals, b = numpy.histogram(self._parent._data.get_slice(slice_num), bins = nbin, range = [mi, ma])      

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
class display(subclass):
    """
    This is a collection of display tools for the raw and reconstructed data
    """
    
    def __init__(self, parent = []):
        subclass.__init__(self, parent)
        
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


    def slice(self, index = None, dim = 0, fig_num = []):
        '''
        Display a 2D slice of 3D volumel
        '''
        self._figure_maker_(fig_num)

        if index is None:
            slice_num = self._parent.data.shape[index] // 2

        img = self._parent.get_slice(index)

        if mirror: img = numpy.fliplr(img)
        if upsidedown: img = numpy.flipud(img)
        
        if self._dynamic_range != []:
            im = plt.imshow(img, cmap = self._cmap, origin='lower', vmin = self._dynamic_range[0], vmax =self._dynamic_range[1])
        else:
            im = plt.imshow(img, cmap = self._cmap, origin='lower')
            
        # Avoid creating colorbar twice:        
        if fig_num is []:
            plt.colorbar(shrink=0.5)            

        plt.show()
        plt.pause(0.0001)

    def slice_movie(self, dim = 1, fig_num = []):
        '''
        Display a 2D slice of 3D volumel
        '''
        self._figure_maker_(fig_num)

        slice_num = 0
        img = self._parent.data.get_slice(slice_num)
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

        img = self._parent.analyse.sum(dim)
        
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
#           PROCESS class
# **************************************************************
        
    class process(subclass):
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
                        self._parent.data[jj] = numpy.array(block > threshold, dtype = 'float32')
                    else:
                        self._parent.data[jj] = numpy.array(block < threshold, dtype = 'float32')

                else:
                    
                    # Setting velues below threshold to zero:
                    if morethan:
                        blk = block[block < threshold] = 0
                    else:
                        blk = block[block > threshold] = 0
                        
                    self._parent.data[jj] = blk

        # add a record to the history:
        self._parent.meta.history.add_record('process.threshold', [threshold, binary, morethan])
        
    def interpolate_holes(self, kernel = [3,3,3], mask2d):
        '''
        Fill in the holes, for instance, saturated pixels.
        
        Args:
            mask2d: holes are zeros.
        '''
        
        prnt = self._parent
        
        for ii in range(prnt.data.length):
                    
            # Compute the filler:
            tmp = ndimage.filters.gaussian_filter(mask2d, sigma = kernel)        
            tmp[tmp > 0] **= -1

            tmp *= ndimage.filters.gaussian_filter(prnt.data[ii] * mask2d, sigma = kernel)
            
            # Inverse the mask:            
            mask2d = ~numpy.bool8(mask2d)
        
            # Apply filler:    
            image = prnt.data[ii] 
            image[mask2d] = tmp[mask2d]

            prnt.data[ii] = image
                
            # Show progress:
            progress_bar(ii / prnt.data.length)
            
        self._parent.meta.history.add_record('process.interpolate_holes(kernel, mask2d)', kernel)

    def residual_rings(self, kernel=[3, 3]):
        '''
        Apply correction by computing outlayers .
        '''
        
        prnt = self._parent
        
        # Compute mean image of intensity variations that are < 5x5 pixels
        print('Our best agents are working on the case of the Residual Rings. This can take years if the kernel size is too big!')

        tmp = prnt.data.empty_slice()
        
        for ii in range(prnt.data.length):                    
            # Compute:
            tmp += prnt.data[ii] - ndimage.filters.median_filter(prnt.data[ii], sigma = kernel) 
            
        tmp /= prnt.data.length
        
        print('Subtract residual rings.')
        
        for ii in range(prnt.data.length):
            prnt.data[ii] = prnt.data[ii] - tmp

            progress_bar(ii / prnt.data.length)

        self._parent.meta.history.add_record('process.residual_rings(kernel)', kernel)
        
        print('Residual ring correcion applied.')

    def subtract_air(self, air_val = None):
        '''
        Subtracts a coeffificient from each projection, that equals to the intensity of air.
        We are assuming that air will produce highest peak on the histogram.
        '''
        
        prnt = self._parent

        print('Air intensity is subtracted')
        
        for ii in range(prnt.data.length):

            if air_val is None:  
                # Take pixels that belong to the 5 pixel-wide margin.
                border = prnt.data[ii]
                border = [border[:10, :], border[-10:, :], border[:, -10:], border[:, :10]
              
                y, x = numpy.histogram(border, 2**12)
                x = (x[0:-1] + x[1:]) / 2
        
                # Subtract maximum argument:    
                val = x[y.argmax()]
                        
                prnt.data[ii] = prnt.data[ii] - val
            else:
                prnt.data[ii] = prnt.data[ii] - air_val

            progress_bar(ii / prnt.data.length)
                    
        self._parent.meta.history.add_record('subtract_air(air_val)', air_val)        
    
    def medipix_quadrant_shift(self):
        '''
        Expand the middle line
        '''
        
        print('Applying medipix pixel shift.')
        
        # this one has to be applied to the whole dataset as it changes its size
        data = self._parent.data.total
        
        data[:,:, 0:data.shape[2]//2 - 2] = data[:,:, 2:data.shape[2]/2]
        data[:,:, data.shape[2]//2 + 2:] = data[:,:, data.shape[2]//2:-2]

        # Fill in two extra pixels:
        for ii in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-ii) < numpy.abs(2-ii)) else 2
            data[:,:, data.shape[2]//2 - ii] = data[:,:, data.shape[2]//2 + closest_offset]

        # Then in columns
        data[0:data.shape[0]//2 - 2,:,:] = data[2:data.shape[0]//2,:,:]
        data[data.shape[0]//2 + 2:, :, :] = data[data.shape[0]//2:-2,:,:]

        # Fill in two extra pixels:
        for jj in range(-2,2):
            closest_offset = -3 if (numpy.abs(-3-jj) < numpy.abs(2-jj)) else 2
            data[data.shape[0]//2 - jj,:,:] = data[data.shape[0]//2 + closest_offset,:,:]

        self._parent.data.total = data

        self._parent.meta.history.add_record('Quadrant shift', 1)
        print('Medipix quadrant shift applied.')

    def skyscan_flat(self):    
        '''
        Apply flat field like they do it in skyscan.
        '''
        prnt = self._parent
        
        if self._parent.meta.geometry.roi_fov:
            print('Object is larger than the FOV!')   
            
        for ii, block in enumerate(prnt.data):
            
            if self._parent.meta.geometry.roi_fov:
                air_values = numpy.ones(block.shape[:2]) * 2**16 - 1
            else:
                air_values = numpy.max(block, axis = 2)
            
            air_values = air_values.reshape((air_values.shape[0],air_values.shape[1],1))
    
            prnt.data[ii] = block / air_values

            progress_bar(ii / prnt.data.length)
        
        # add a record to the history:
        self._parent.meta.history.add_record('process.skyscan_flat', 1)

        print('Skyscan flat field correction applied.')

    def flat_field(self):
        '''
        Apply flat field correction based on reference images.
        '''
        prnt = self._parent
        
        for ii in range(prnt.data.length):
            
            prnt.data[ii] = block / air_values    
            
            ref = self._parent.data.get_ref(proj_num = ii)
                
            if numpy.min(ref) <= 0:
                self._parent.warning('Flat field reference image contains zero (or negative) values! Will replace those with little tiny numbers.')

                tiny = ref[ref > 0].min()
                ref[ref <= 0] = tiny

            # If there is a dark field, use it.
            if not prnt._dark is None:    
                prnt.data[ii] = prnt.data[ii] - prnt._dark

                # Is dark subtracted from ref images internally?
                ref = ref - prnt._dark
                
            # Use flat field:
            prnt.data[ii] /= ref    

            progress_bar(ii / prnt.data.length)
            
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
        
        for ii in range(prnt.data.length):
        
            if (air_intensity != 1.0):
                prnt.data[ii] /= air_intensity
            
            # Apply a bound to large values:
            prnt.data[ii] = numpy.clip(prnt.data[ii], a_min = numpy.exp(-upper_bound), a_max = numpy.exp(-lower_bound))
            
                  
            # In-place negative logarithm
            prnt.data[ii] = -numpy.log(prnt.data[ii])
            
            # Progress:
            progress_bar(ii / prnt.data.length)
                    
        self._parent.message('Logarithm is applied.')
        self._parent.meta.history.add_record('process.log(air_intensity, bounds)', [air_intensity, lower_bound, upper_bound])

    def salt_pepper(self, kernel = [3, 3]):
        '''
        Get rid of nasty speakles
        '''
        
        prnt = self._parent
        
        for ii in range(prnt.data.length):    
            # Make a smooth version of the data and look for outlayers:
            smooth = ndimage.filters.median_filter(prnt.data[ii], kernel)
            mask = prnt.data[ii] / smooth
            mask = (numpy.abs(mask) > 1.5) | (numpy.abs(mask) < 0.75)

            img = prnt.data[ii]
            img[mask] = smooth[mask]
            prnt.data[ii] = img

            progress_bar(ii / prnt.data.length)

        self._parent.message('Salt and pepper filter is applied.')

        self._parent.meta.history.add_record('process.salt_pepper(kernel)', kernel)

    def add_noise(self, std, mode = 'gaussian'):
        '''
        Noisify the projection data!
        '''
        from skimage import util
        
        self._parent.data.total = util.random_noise(self._parent.data.total, mode = mode, var = std**2)
        
        self._parent.meta.history.add_record('process.add_noise(std, mode)', [std, mode])
        
    def simple_tilt(self, tilt):
        '''
        Tilts the sinogram
        '''
        
        prnt = self._parent
        
        print('Applying tilt.')
        for ii in range(prnt.data.length):     
            prnt.data[ii] = interp.rotate(prnt.data[ii], -tilt, reshape=False)
            progress_bar(ii / prnt.data.length)
            
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
            
        if bottom_right[1] > 0:
            self._parent.data._data = self._parent.data._data[top_left[1]:-bottom_right[1], :, :]
        else:
            self._parent.data._data = self._parent.data._data[top_left[1]:, :, :]

        if bottom_right[0] > 0:
            self._parent.data._data = self._parent.data._data[:, :, top_left[0]:-bottom_right[0]]
        else:
            self._parent.data._data = self._parent.data._data[:, :, top_left[0]:]

        self._parent.data._data = numpy.ascontiguousarray(self._parent.data._data, dtype=numpy.float32)
        gc.collect()
        
        if numpy.min(self._parent.data._data.shape) == 0:
            self._parent.warning('Wow! We have just cropped the data to death! No data left.')

        self._parent.meta.history.add_record('process.ccrop(top_left, bottom_right)', [top_left, bottom_right])

        self._parent.message('Sinogram cropped.')
        
    def equivalent_thickness(self, energy, spectrum, compound, density):
        '''
        Transfer intensity values to equivalent thicknessb
        '''
        
        # Assuming that we have log data!
        if not 'process.log(air_intensity, bounds)' in self._parent.meta.history.keys:                        
            self._parent.error('Logarithm was not found in history of the projection stack. Apply log first!')
        
        print('Generating the transfer function.')
        
        # Attenuation of 1 mm:
        mu = simulate.spectra.linear_attenuation(energy, compound, density, thickness = 0.1)
        width = self._parent.data.shape[2]

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
        
        for ii in range(prnt.data.length):  
            
            prnt.data[ii] = numpy.array(numpy.interp(prnt.data[ii], synth_counts, thickness), dtype = 'float32')
            progress_bar(ii / prnt.data.length)
                                              
    def bin_projections(self):
        '''
        Bin data with a factor of two in detector plane.
        '''
        print('Binning projection data.')    
        
        data = self._parent.data.total
        data[:, :, 0:-1:2] += data[:, :, 1::2]
        data = data[:, :, 0:-1:2] / 2

        data[0:-1:2, :, :] += data[1::2, :, :]
        data = data[0:-1:2, :, :] / 2

        self._parent.data.total = data

        # Multiply the detecotr pixel width and height:
        self._parent.meta.geometry.det_pixel[0] = self._parent.meta.geometry.det_pixel[0] * 2
        self._parent.meta.geometry.det_pixel[1] = self._parent.meta.geometry.det_pixel[1] * 2