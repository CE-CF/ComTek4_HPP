import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft,irfft
import scipy as sp
from PIL import Image
import matplotlib.cm as cm
from scipy.fftpack import dct, idct
import math
from time import time
from sys import argv
from itertools import chain

def _time(f):
    def wrapper(*args):
        start = time()
        r = f(*args)                            
        end = time()
        print("%s timed %f" % (f.__name__, end-start) )
        return r
    return wrapper

def show(im, text, imgtype=0):
    fig = plt.figure()
    if (imgtype == 1):
        plt.imshow(im,cmap='gray',vmax = np.max(im)*0.01,vmin = 0)
    else:
        plt.imshow(im, cmap=plt.cm.Greys_r)

    plt.title(text)
    plt.colorbar()
    plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
    plt.show()

@_time
def reshape_split(image: np.ndarray, kernel_size: tuple):

    img_height, img_width = image.shape #, channels
    tile_height, tile_width = kernel_size
    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width#,  #channels
                                )
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array

def dct2(Array):                    # Array[P][Q][M][N]
    P = len(Array)              
    Q = len(Array[0])       
    M = len(Array[0][0])
    N = len(Array[0][0][0])
    outMatrix = np.empty((P,Q,M,N),dtype='int16')
    for p in range(P):
        for q in range(Q):
            for m in range(M):
                for n in range(N):
                    outMatrix = Array[p][q][m][n]

    


imageX = 800
imageY = 600
tilesize = (8,8)
img = np.random.randint(0,255,(imageX,imageY), dtype='int16')
img -= 128
tiles = reshape_split(img, tilesize)
print(tiles[50][50])


