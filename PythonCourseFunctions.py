#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 21:20:54 2019

@author: grael
"""

# Standard Libraries
import os
import sys
import itertools
import copy
import random
import math
import time


# The most important non-standard libraries
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


# image processing!!
import skimage
from skimage import morphology
from PIL import Image


# a library (advanced use; usually not needed for most things) for speed
from numba import jit

# UMAP: conda install -c conda-forge umap-learn
import umap

"""Examples:
    vanilla python!
"""

# strings.
# strings are enclosed in quotes.
x = 'abc123'
print(x)
x
x = "abc123"
print(x)
# note the use of "" to create a string that has a ' in it.
y = "Wouldn't you like to program in Python?"
print(y)
# special functional strings (advanced python 3 feature)
y = f'abc123{x}'
print(y)

# integers and floats.
print(type(2), type(2.))

# lists.
x = [2]
print(type(x))
print(type(x[0]))

# sets.
x = {2}
print(type(x))
try:
    print(type(x[0]))
except:
    print('oops! error.')
y = {3}
print(x & y)

# dictionaries.
x = {'x_key1': 2, 'x_key2': 3}
print(type(x))
print(type(x['x_key1']))
print(len(x))
print(x.keys())

# operations
print(1+1)
print(1+1.)
print(3 / 2)
print(3 / 2.)
# floor division: 2 goes into 3 once.
print(3 // 2)
# remainder: 3 divided by 2 gives remainder 1.
print(3 % 2)


# object values and references
x = 2
y = x
print(y)
x = 3
print(y)



# logic
x = True
y = False
print(x and y)
print(x or y)
print(x and not y)
print(not x and y)
print(not (x and y))


# functions
def my_first_function(x, y):
    return x+y
print(my_first_function(1, 2))


# some objects have attributes or functions, called methods, attached to them.
# methods and attributes are called/accessed using a .
x = {1, 2, 3}
y = {2, 3, 4}
# example of set methods.
print(x.intersection(y))
print(x.union(y))
# calling the method without any arguments returns the method argument
print(type(x.intersection))
i = x.intersection
print(type(i))
print(i(y))

x = 'abcd_efg_lnmop'
y = x.split('_')
print(y)
print('+'.join(y))


"""Examples:
    os
"""
root = '/Users/grael/Desktop'
try:
    files = os.listdir(root)
    print(files)
    
    # make a new folder if it doesn't already exist
    new_dir = os.path.join(root, 'New Directory')
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
        
    # delete it
    # be very careful with this... you can delete important files if you accidentally remove the wrong directory
    # and there is no safety switch in python unless you make it yourself
    new_dir_files = os.listdir(new_dir)
    if len(new_dir_files) == 0:
        os.rmdir(new_dir)
    else:
        print('I found a file in this directory. Not deleting...')
        
except:
    print(f"couldn't create directory at {root}. This root directory may not exist. try changing the root variable.")
    

"""Examples:
    sys
"""
if False:
    sys.exit()
else:
    print("This would have closed the program, but it won't execute.")


"""Examples:
    itertools
"""
x = [1, 'a', (1, 2, 3)]
print(list(itertools.combinations(x, r=2)))


"""Examples:
    copy
"""
x = 2
y = 3
z = 'abc'
l = [x, y, z]

c = copy.copy(l)
dc = copy.deepcopy(l)

print(c)
print(dc)

x = 'New Value'

print(c)
print(dc)


"""Examples:
    random
"""
x = [1, 2, 3, 4, 5]
c = random.choice(x)
print(c)
print(x)

# note that this is an in-place operation; the value of x is modified!
random.shuffle(x)
print(x)



"""Examples:
    math
"""
print(math.pi)

# note that this is only approximately equal to zero, due to approximations.
print(math.sin(math.pi))



"""Examples:
    time
"""
def time_my_function(func, args):
    """
    func is a function object.
    args is a tuple of arguments.
    """
    assert type(args) == tuple
    
    start_time = time.time()
    
    # the * in front of args is an operator that does something called "tuple unpacking". its handy.
    func_out = func(*args)
    
    end_time = time.time()
    elapsed_time = end_time  - start_time
    
    return func_out, elapsed_time
    


"""Examples:
    numpy
"""
# generate some data
z = np.zeros(shape=(100, 3), dtype=bool)
n = np.full(4, np.nan)
e = np.empty(shape=(6,6))

r = np.random.rand(10, 2)
rp = np.random.permutation(r)

group1 = np.random.randn(10000)
group2 = np.random.randn(10000)

# note that arrays have some methods built in.
# when calling a method with no argumants, you still have to use ().
r.mean()
r.mean(axis=0)
r.mean(axis=1)
r.std()

# I promise, this stuff will cause you to lose sleep someday.
print(r.shape)
print(np.reshape(r, (-1,1)).shape)
print(np.reshape(r, (1,-1)).shape)
print(r[:, :, None].shape)
print(r[:, None].shape)

# multiplication
print(r * 2)
print(r * r)

# elementwise minimum of two arrays with the same shape
a1 = np.random.rand(10, 4)
a2 = np.random.rand(10, 4)
# note that arrays have some methods
print(a1.mean(), a2.mean())
print(np.minimum(a1, a2).mean())

# matrix transpose
print(r.T.shape)
print(r[:,:,None].T.shape)

# matrix multiplication
print(np.dot(r, r.T))
print(np.dot(r.T, r))


"""Examples:
    scipy
"""
print(stats.ttest_ind(group1, group2))

group1 += 1
print(stats.ttest_ind(group1, group2))

print(r.shape)
pairwise_distances = sp.spatial.distance.pdist(r)
print(pairwise_distances)

r_new = np.random.rand(10, 3)
# 3D points with pairwise euclidean distance
pairwise_distances_new = sp.spatial.distance.pdist(r_new)
print(pairwise_distances_new)


"""Examples:
    matplotlib
"""
plt.figure()
plt.scatter(r[:,0], r[:,1])
plt.show()

plt.figure(dpi=100)
plt.scatter(
        r[:,0],
        r[:,1],
        c=np.random.randint(0, r.shape[0], r.shape[0]),
        cmap='coolwarm',
        s=800,
        alpha=0.6)
plt.show()

# ugly!
plt.figure()
plt.plot(group1, group2)
plt.show()

"""
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
Break, review, discuss, whatever.
"""

"""Examples:
    for
    while
"""
x = list(range(10))
print(x)
for x in range(10):
    print(x)
for x in list(range(10)):
    print(x)

for x in 'abcdefg':
    print(x)

x = [a.mean() for a in [np.random.rand(100) for _ in range(10)]]
print(x)

x = [p for p in 'abcdefg' if p in ['g', 'c']]
print(x)

def while_loop_example(max_iter):
    
    choices = np.arange(100)
    count = 0
    total = 0
    while True:
        total += 1
        r = np.random.choice(choices)
        if r == 2:
            count += 1
        if total % 10000 == 0:
            print(total)
        if count >= max_iter:
            break
        else:
            continue
    print(2)
    return count
        
x = 2

print(time_my_function(while_loop_example, (200,)))

"""Examples:
    pandas
"""
data_dict = {
        'Name': [],
        'Age': [],
        'Sex': [],
        'Programmer': []}

for i in range(1000):
    name = np.random.randint(0, 1000000)
    Age =  np.random.randint(10, 100)
    Sex = random.choice(['Male', 'Female', 'Other'])
    programmer = random.choice([True, False])
    
    data_dict['Name'].append(name)
    data_dict['Age'].append(Age)
    data_dict['Sex'].append(Sex)
    data_dict['Programmer'].append(programmer)
    
df = pd.DataFrame(data_dict)

print(df.head())

print(df.groupby('Sex').mean()['Programmer'])

"""Examples:
    seaborn
"""

for kind in ['swarm', 'box', 'violin']:
    plt.figure()
    sns.catplot(
            data=df,
            x='Sex',
            col='Programmer',
            y='Age',
            kind=kind)
    plt.show()





"""Examples:
    numba + everything else!
"""
@jit
def mandelbrot(z,maxiter,horizon,log_horizon):
    c = z
    for n in range(maxiter):
        az = abs(z)
        if az > horizon:
            return n - np.log(np.log(az))/np.log(2) + log_horizon
        z = z*z + c
    return 0

@jit
def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    horizon = 2.0 ** 40
    log_horizon = np.log(np.log(horizon))/np.log(2)
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width,height))
    for i in range(width):
        for j in range(height):
            n3[i,j] = mandelbrot(r1[i] + 1j*r2[j],maxiter,horizon, log_horizon)
    return (r1,r2,n3)


def mandelbrot_image(xmin, xmax, ymin, ymax, width=10, height=10, maxiter=256, cmap='jet', gamma=0.3):
    """you can fiddle with all the parameters in here and see what happens..."""
    dpi = 100
    img_width = dpi * width
    img_height = dpi * height
    x, y, z = mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height, maxiter)

    fig, ax = plt.subplots(figsize=(width, height), dpi=72)
    ticks = np.arange(0, img_width, 3 * dpi)
    x_ticks = xmin + (xmax - xmin) * ticks / img_width
    plt.xticks(ticks, x_ticks)
    y_ticks = ymin + (ymax - ymin) * ticks / img_width
    plt.yticks(ticks, y_ticks)
    ax.set_title(cmap)

    norm = mpl.colors.PowerNorm(gamma)
    ax.imshow(z.T, cmap=cmap, origin='lower', norm=norm)

# p = [-0.8,-0.7,0,0.1]
p = [-0.73793,-0.736,0.1885,0.19]
s = 1
mandelbrot_image(p[0]*s,p[1]*s,p[2]*s,p[3]*s, cmap='hot', maxiter=2048)
plt.show()


"""Examples:
    Image processing and some fun stuff
"""
image_file = '/Users/grael/Desktop/cat.jpg'
with Image.open(image_file) as f:
    cat = np.array(f)
    
plt.figure(dpi=300)
plt.imshow(cat)
plt.show()


def show_transformed_cat(cat, func):
    f = func(cat)
    r = skimage.exposure.rescale_intensity(f, out_range=np.uint8)
    plt.figure(dpi=180)
    plt.imshow(r.astype(np.uint8))
    plt.show()

float_cat = cat.astype(np.float64)
show_transformed_cat(float_cat, np.arcsinh)
show_transformed_cat(float_cat, np.exp)


"""
https://www.synapse.org/#!Synapse:syn17813510
"""
def show_me(image, dpi):
    plt.figure(dpi=dpi)
    plt.imshow(image, cmap='gray')
    plt.show()
    
    
image_file = '/Users/grael/Desktop/TONSIL-1_40X_10.tif'
try:
    with Image.open(image_file) as f:
        cycif = np.array(f)
except:
    cycif = skimage.io.imread(image_file)
show_me(cycif, 180)

cycif_roi = cycif[2000:3000, 3000:5000]
show_me(cycif_roi, 180)

t = np.percentile(cycif_roi[cycif_roi>0], 98)
t_cycif_roi = cycif_roi > t
show_me(t_cycif_roi, 180)

labeled = skimage.measure.label(t_cycif_roi)
show_me(labeled, 180)

cleaned = morphology.remove_small_objects(labeled, area_threshold=50)
show_me(cleaned, 180)

rp = skimage.measure.regionprops(
        cleaned,
        intensity_image=cycif_roi,
        coordinates='rc')

properties = [
        'label',
        'area',
        'perimeter',
        'eccentricity',
        'extent',
        'mean_intensity',
        'min_intensity',
        'max_intensity']


rp_dict = {p: [r[p] for r in rp] for p in properties}

rp_df = pd.DataFrame(rp_dict, index=rp_dict['label']).drop(columns=['label'])

umap_model = umap.UMAP()
umap_coords = umap_model.fit_transform(rp_df.drop(columns=['mean_intensity', 'min_intensity', 'max_intensity']))


plt.figure(dpi=200)
g = sns.scatterplot(umap_coords[:,0], umap_coords[:,1], hue=rp_df['mean_intensity'])
g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
plt.show()