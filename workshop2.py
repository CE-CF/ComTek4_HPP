#!/usr/bin/python3
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

def _time(f):
    def wrapper(*args):
        start = time()
        r = f(*args)
        end = time()
        print("%s timed %f" % (f.__name__, end-start) )
        return r
    return wrapper


@_time
def for_loop_split(image: np.ndarray, kernel_size: tuple):
	print(image.shape)
	img_height, img_width = image.shape # , channels
	tile_height, tile_width = kernel_size

	tiled_array = np.zeros((img_height // tile_height,
                            img_width // tile_width,
                            tile_height,
                            tile_width#,
                            #channels
                            ))

	y = x = 0
	for i in range(0, img_height, tile_height):
		for j in range(0, img_width, tile_width):
			tiled_array[y][x] = image[i:i+tile_height,
                                      j:j+tile_width#,
                                      #:channels
                                      ]
			x += 1
		y += 1
		x = 0

	return tiled_array


@_time
def stride_split(image: np.ndarray, kernel_size: tuple):
    # Image & Tile dimensions
    img_height, img_width = image.shape #, channels
    tile_height, tile_width = kernel_size

    # bytelength of a single element
    bytelength = image.nbytes // image.size

    tiled_array = np.lib.stride_tricks.as_strided(
        image,
        shape=(img_height // tile_height,
               img_width // tile_width,
               tile_height,
               tile_width,
              # channels
              ),
        strides=(img_width*tile_height*bytelength,#*channels,
                 tile_width*bytelength,#*channels,
                 img_width*bytelength,#*channels,
                 bytelength,#*channels,
                 #bytelength
                 )
    )
    return tiled_array


@_time
def reshape_split(image: np.ndarray, kernel_size: tuple):

    img_height, img_width = image.shape #, channels
    tile_height, tile_width = kernel_size

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width#,
                                #channels
                                )
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float64)
    rgb[:,:,[1,2]] -= 128
    return np.uint8(rgb.dot(xform.T))

def close_figure(event):
    if event.key == 'escape':
        plt.close(event.canvas.figure)

def show(im):
	fig = plt.figure()
	plt.imshow(im, cmap=plt.cm.Greys_r)
	plt.colorbar()
	plt.gcf().canvas.mpl_connect('key_press_event', close_figure)
	plt.show()


img = np.asarray(Image.open("Faur2.png").convert('L'))
#img = np.asarray(Image.open("Faur2.png").convert('RGB'))

#img = rgb2ycbcr(img)
show(img)

t1, t2 = (argv[1], argv[2])
tilesize = (int(t1), int(t2))

tiles_1 = for_loop_split(img, tilesize)
tiles_2 = stride_split(img, tilesize)
tiles_3 = reshape_split(img, tilesize)

if (tiles_1 == tiles_2).all() and (tiles_2 == tiles_3).all():
    n = tiles_1.shape[0] * tiles_1.shape[1]
    print("\nAll tile arrays are equal.")
    print("Each array has %d tiles\n" % (n))



@_time
def dct2(block):
    return dct(dct(block, axis=0), axis=1)
@_time
def idct2(block):
    return idct(idct(block, axis=0), axis=1)

def dctOurs(x):
    N = len(x)
    x2 = np.empty(2*N,float)
    x2[:N] = x[:]
    x2[N:] = x[::-1]

    X = rfft(x2)									# This function computes the one-dimensional n-point discrete Fourier Transform (DFT) of a real-valued array by means of an efficient algorithm called the Fast Fourier Transform (FFT).
    phi = np.exp(-1j*math.pi*np.arange(N)/(2*N))
    return np.real(phi*X[:N])

@_time
def dct2Ours(block):
    M = block.shape[0]
    N = block.shape[1]
    a = np.empty([M,N],float)
    X = np.empty([M,N],float)

    for i in range(M):
        a[i,:] = dctOurs(block[i,:])
    for j in range(N):
        X[:,j] = dctOurs(a[:,j])

    return X



#img = ycbcr2rgb(img)
#show(img)
dctmatrix = [[0 for x in range(len(tiles_1))]for x in range(len(tiles_1[0]))]
for x in range(len(tiles_1)):
	for y in range(len(tiles_1[0])):
		dctmatrix[x][y] = dct2Ours((tiles_1[x][y]))

print(f'The original matrix: \n {tiles_1[50][50]}\n')

print(f'The dctmatrix: \n {dctmatrix[50][50]}\n')
"""
Source: https://www.math.cuhk.edu.hk/~lmlui/dct.pdf

There exist a standardizezd quantization matrix where values range from 0-100, 100 is high quality and low compression and 0 is high compression and low quality
Below is the 50 version
"""

Q_50 = [[16,11,10,16,24,40,51,61],
		[12,12,14,19,26,58,60,55],
		[14,13,16,24,40,57,69,56],
		[14,17,22,29,51,87,80,62],
		[18,22,37,56,68,109,103,77],
		[24,35,55,64,81,104,113,92],
		[49,64,78,87,103,121,120,101],
		[72,92,95,98,112,100,103,99]]

"""
For a quality greater than 50, the standard quantization matrix is multiplied by 
(100-quality level)/50. For a quality level less than 50(more compression, lower image quality),
the standard quantization matrix is multiplied by 50/quality level.
The scaled quantization matrix is then rounded and clipped to have positive integer values ranging from 1 to 255.
"""

"""
Quantization is achived by dividing each element in the transformed image matrix D by the corresponding element in the quantization matrix, and then rounding to the nearest integer value.
for this example the Q_50 matrix is used.
"""
quantmatrix = [[0 for x in range(len(tiles_1))]for x in range(len(tiles_1[0]))]
for x in range(len(tiles_1)):
	for y in range(len(tiles_1[0])):
		 quantmatrix[x][y] = np.round((dctmatrix[x][y]/Q_50), 0)

print(f'The quantmatrix: \n {quantmatrix[50][50]}')