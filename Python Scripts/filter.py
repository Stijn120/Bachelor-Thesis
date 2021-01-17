# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:30:01 2020

@original author: Joeri Hartjes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from astropy.convolution import Gaussian2DKernel
from tqdm import tqdm

def normalize(x):
        xmax, xmin = x.max(), x.min()
        x = (x-xmin)/(xmax-xmin)
        return x

def blob(size,size_noise,shape_noise, phos_size_min):
        if size_noise and shape_noise:
                shape_noise_range_max = int(size/4)
                shape_noise_range_min = int(phos_size_min/4)
                
                rand_theta = np.random.uniform(0,6.27,size=(1))
                rand_cov1 = np.random.randint(shape_noise_range_min,shape_noise_range_max,size=(1))
                rand_cov2 = np.random.randint(shape_noise_range_min,shape_noise_range_max,size=(1))
                gaussian_2D_kernel = Gaussian2DKernel(rand_cov1,rand_cov2,theta=rand_theta,x_size=size,y_size=size,mode='center')
              
                kernel = gaussian_2D_kernel.array
                kernel = normalize(kernel)
                return np.swapaxes(np.array([kernel,kernel,kernel]),0,2)
        if size_noise and not shape_noise:
                shape_noise_range_max = int(size/4)
                shape_noise_range_min = int(phos_size_min/4)
                
                rand_cov1 = np.random.randint(shape_noise_range_min,shape_noise_range_max,size=(1))
                gaussian_2D_kernel = Gaussian2DKernel(rand_cov1,rand_cov1,theta=0,x_size=size,y_size=size,mode='center')
              
                kernel = gaussian_2D_kernel.array
                kernel = normalize(kernel)
                return np.swapaxes(np.array([kernel,kernel,kernel]),0,2)
        if not size_noise and not shape_noise:
                blob_size = int(size/4)
                gaussian_2D_kernel = Gaussian2DKernel(blob_size,blob_size,theta=0,x_size=size,y_size=size,mode='center')
                kernel = gaussian_2D_kernel.array
                kernel = normalize(kernel)
                return np.swapaxes(np.array([kernel,kernel,kernel]),0,2)
        
def imageplot(grid, title="Image"):
    plt.figure(1)
    plt.imshow(grid)
    plt.title(title)
    plt.axis("off")
    plt.show()
    
class grid:
        def __init__(self, grid_res, sim_res, spacing, phos_size_max):
                self.phos_size_max = phos_size_max
                self.spat_noise = False
                self.grid_res = grid_res
                self.sim_res = sim_res
                self.spacing = spacing 
                ticks = np.linspace(0, sim_res+(spacing*3) - 1, grid_res+4).astype("uint16")
                self.grid_max = ticks[-1]+1
                self.electrode_grid = np.zeros((self.grid_max, self.grid_max), "bool")
                self.ticks = ticks[2:-2]
                for tick in self.ticks:
                        self.electrode_grid[tick,self.ticks] = True  
        
        def add_spatial_noise(self, spat_nr, spat_nsdv):
                xn = truncnorm(a=-spat_nr/spat_nsdv, b=+spat_nr/spat_nsdv, scale=spat_nsdv).rvs(size=self.grid_res*self.grid_res)
                xn = xn.round().astype(int)
                self.xn = xn.reshape((self.grid_res,self.grid_res))
                yn = truncnorm(a=-spat_nr/spat_nsdv, b=+spat_nr/spat_nsdv, scale=spat_nsdv).rvs(size=self.grid_res*self.grid_res)
                yn = yn.round().astype(int)
                self.yn = yn.reshape((self.grid_res,self.grid_res))
                self.spat_noise = True
                self.electrode_grid = np.zeros((self.grid_max, self.grid_max), "bool")
                self.locations = []
                for x in range(len(self.xn)):
                        for y in range(len(self.yn)):
                                self.locations.append(((self.ticks[x]+self.xn[x,y]),(self.ticks[y]+self.yn[x,y])))
                                self.electrode_grid[self.ticks[x]+self.xn[x,y],self.ticks[y]+self.yn[x,y]] = True
                                
        def plot_filter(self):   
                x,y = np.where(self.electrode_grid)
                if (size_noise and shape_noise) or (size_noise and not shape_noise):
                        for i in tqdm(range(len(x))):
                                phos = self.phosphenes[i]
                                patchx,patchy,channels = phos.shape
                                half = int(patchx/2)
                                x_ind = x[i]
                                y_ind = y[i]
                                self.phosphene_image[x_ind-half:x_ind+half,y_ind-half:y_ind+half,:] = np.add(self.phosphene_image[x_ind-half:x_ind+half,y_ind-half:y_ind+half,:], phos)
                                self.phosphene_image[self.phosphene_image > 1] = 1
                if not size_noise and not shape_noise: 
                        phos = self.phosphenes[0]
                        patchx,patchy,channels = phos.shape
                        half = int(patchx/2)
                        for i in tqdm(range(len(x))):
                                x_ind = x[i]
                                y_ind = y[i]
                                self.phosphene_image[x_ind-half:x_ind+half,y_ind-half:y_ind+half,:] = np.add(self.phosphene_image[x_ind-half:x_ind+half,y_ind-half:y_ind+half,:], phos)
                                self.phosphene_image[self.phosphene_image > 1] = 1
                #imageplot(self.phosphene_image)
                
        def plot_phosphene_image(self, binary_image):
                count = 0
                if (size_noise and shape_noise) or (size_noise and not shape_noise):
                        for i in range(len(binary_image)):
                                for j in range(len(binary_image[0])):
                                        if binary_image[i,j]:
                                                locx, locy = self.locations[count]
                                                phos = self.phosphenes[count]
                                                patchx,patchy,channels = phos.shape
                                                half = int(patchx/2)
                                                self.phosphene_image[locx-half:locx+half,locy-half:locy+half,:] = np.add(self.phosphene_image[locx-half:locx+half,locy-half:locy+half,:], phos)
                                                self.phosphene_image[self.phosphene_image > 1] = 1
                                        count +=1        
                if not size_noise and not shape_noise:
                        phos = self.phosphenes[0]
                        patchx,patchy,channels = phos.shape
                        half = int(patchx/2)
                        for i in range(len(binary_image)):
                                for j in range(len(binary_image[0])):
                                        if binary_image[i,j]:
                                                locx, locy = self.locations[count]
                                                self.phosphene_image[locx-half:locx+half,locy-half:locy+half,:] = np.add(self.phosphene_image[locx-half:locx+half,locy-half:locy+half,:], phos)
                                                self.phosphene_image[self.phosphene_image > 1] = 1
                                        count +=1
                #imageplot(self.phosphene_image)
                return self.phosphene_image
                
        def phosphenes(self, size_noise, shape_noise, phos_size_min, color_noise_p,color_noise_i):
                self.size_noise = size_noise
                self.shape_noise = shape_noise
                x,y = np.where(self.electrode_grid)
                self.phosphene_image = np.zeros((len(self.electrode_grid),len(self.electrode_grid),3))
                self.phosphenes = []
                
                if (size_noise and shape_noise) or (size_noise and not shape_noise):
                        for i in tqdm(range(len(x))):
                                phosphene = blob(self.phos_size_max,size_noise,shape_noise, phos_size_min)
                                sample = np.random.uniform(0,1)
                                if sample < color_noise_p:
                                        multi = np.random.uniform(color_noise_i,1,3)
                                        for i in range(len(multi)):
                                                phosphene[:,:,i] = phosphene[:,:,i] * multi[i]
                                self.phosphenes.append(phosphene)
                                
                if not size_noise and not shape_noise: 
                        phosphene = blob(self.phos_size_max,size_noise,shape_noise, phos_size_min)
                        self.phosphenes.append(phosphene)
        
        def save_noise(self, filename_grid, filename_phosphenes, filename_locations):
                np.save(filename_grid, self.electrode_grid)
                np.save(filename_phosphenes, self.phosphenes)
                np.save(filename_phosphenes, [self.xn,self.yn])
                
        
        def load_noise(self, filename_grid, filename_phosphenes, filename_locations):
                self.electrode_grid = np.load(filename_grid)
                self.phosphenes = np.load(filename_phosphenes)
                noise = np.load(filename_locations)
                self.xn = noise[0]
                self.yn = noise[1]


 # PARAMETERS
# resolution specifications
grid_res = 63                   # amount of phosphenes = grid_res * grid_res
spacing = 32                     # spacing between phosphenes                   
sim_res = grid_res * spacing     

# positional noise
spat_nr = 10                    # spatial noise range (max value = spacing)
spat_nsdv = 7.                 # spatial noise sdv

# phosphene size noise
phos_size_max = 40              # use even number
phos_size_min = 20              # phosphenes can be this times amount smaller/ bigger than each other 

# noise booleans
size_noise = True              # if false, phos_size_max will be phosphene size.
shape_noise = True    

# color noise
color_noise_p = 0.05              # chance for a phosphene to be colored 
color_noise_i = 0.6             #color noise intensity, 0 is higher, 1 = none     

gridfile = 'grid.npy'
phosphenefile = 'phosphenes.npy'   
locationsfile = 'locations.npy'

testpic_file = 'full.txt'
testpic = np.loadtxt(testpic_file, dtype='int')

binary_image = np.random.randint(2, size=(grid_res,grid_res))   #random binary image

grid = grid(grid_res, sim_res, spacing, phos_size_max)
grid.add_spatial_noise(spat_nr, spat_nsdv)

grid.phosphenes(size_noise, shape_noise, phos_size_min, color_noise_p, color_noise_i)

#grid.plot_filter()                             # use this to plot the whole filter
#imageplot(grid.plot_phosphene_image(testpic))     # use this if you want to plot binary image (e.g. '0.txt')

grid.save_noise(gridfile, phosphenefile, locationsfile)

plt.figure(1)
plt.imshow(grid.plot_phosphene_image(testpic))
plt.axis("off")

outputname = "full.png"
plt.savefig(outputname, format='png')

def createPhosheneImage(image):
        phosphene_image = grid.plot_phosphene_image(image)
        return phosphene_image