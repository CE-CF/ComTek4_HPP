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
from itertools import chain

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


img = np.asarray(Image.open("Faur2.png").convert('L'), dtype='int16')
#img = np.asarray(Image.open("Faur2.png").convert('RGB'))

#img = rgb2ycbcr(img)
show(img, "start")
img -= 128

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
    return dct(dct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')

@_time
def dct3(block):
    return dct(dct(block.T, type=3, norm='ortho').T, type=3, norm='ortho')

#@_time
def dctOurs(x):
    N = len(x)
    x2 = np.empty(2*N,float)
    x2[:N] = x[:]
    x2[N:] = x[::-1]

    X = rfft(x2)
    phi = np.exp(-1j*math.pi*np.arange(N)/(2*N))
    return np.real(phi*X[:N])

#@_time
def dct2Ours(x):
    M = x.shape[0]
    N = x.shape[1]
    a = np.empty([M,N],float)
    X = np.empty([M,N],float)

    for i in range(M):
        a[i,:] = dctOurs(x[i,:])
    for j in range(N):
        X[:,j] = dctOurs(a[:,j])

    return X

@_time
def dctTheMatrix(matrix, outMatrix):
    for x in range(len(tiles_1)):
        for y in range(len(tiles_1[0])):
            outMatrix[x][y] = dct2Ours((matrix[x][y]))
    return outMatrix

@_time
def quantTheMatrix(matrix, q_s, outMatrix):
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
             outMatrix[x][y] = np.round((matrix[x][y]/q_s), 0)
    return outMatrix

def rescaleQuant(q50, scale, outMatrix):
    for x in range(len(q50)):
        for y in range(len(q50[0])):
            outMatrix[x][y] = math.floor((scale*q50[x][y] + 50) / 100);
    return outMatrix

@_time
def flattenMatrix(twoDmatrix, outputList):
    #print(f'twoDmatrix = {twoDmatrix}\n')
    outputList.extend(list(np.concatenate(twoDmatrix).flat))
    #print(f'outputList = {outputList}')
    return outputList    


@_time
def reQuantTheMatrix(matrix, q50, outMatrix):
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            outMatrix[x][y] = np.round((matrix[x][y]*Q_50), 0)
    return outMatrix

@_time
def reshape_recompile(matrix):
    matrix = matrix.swapaxes(1, 2)
    newmatrix = matrix.reshape(-1, 800)


    return newmatrix


#img = ycbcr2rgb(img)
#show(img)
print(f'biggest value of original matrix: {np.amax(tiles_1)}\n')
print(f'The original matrix: \n {tiles_1[50][50]}\n')

dctmatrix = np.empty((len(tiles_1),len(tiles_1[0]),len(tiles_1[0][0]), len(tiles_1[0][0][0])), dtype='int16')
"""
dctTheMatrix(tiles_1, dctmatrix)
comparison = dctmatrix == dct2(tiles_1)
if (comparison.all()):
    print(f'Du er dygtig')
else:
    print(f'Det er ikke det samme')
"""
dctmatrix = dct2(tiles_1)
print(f'biggest value of dctmatrix: {np.amax(dctmatrix)}\n')
print(f'The dctmatrix: \n {dctmatrix[50][50]}\n')

show(reshape_recompile(dctmatrix), "dctmatrix", 1)
"""
Source: https://www.math.cuhk.edu.hk/~lmlui/dct.pdf

There exist a standardizezd quantization matrix where values range from 0-100, 100 is high quality and low compression and 0 is high compression and low quality
Below is the 50 version
"""
Q = 80

Q_50 = np.array([
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99]
])

if (Q < 50):
    S = 5000/Q;
else:
    S = 200 - 2*Q;
Q_S = np.empty((len(Q_50),len(Q_50[0])), dtype='int16')
Q_S = rescaleQuant(Q_50, S, Q_S)
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
quantmatrix = np.empty((len(tiles_1),len(tiles_1[0]),len(tiles_1[0][0]), len(tiles_1[0][0][0])), dtype='int16')
quantmatrix = quantTheMatrix(dctmatrix, Q_S, quantmatrix)
print(f'biggest value of quantmatrix: {np.amax(quantmatrix)}\n')
print(f'The quantmatrix: \n {quantmatrix[50][50]}\n')
show(reshape_recompile(quantmatrix), "quantmatrix", 1)
"""
# Everything in here is used to compress the image further
zigZagList = []
flattenMatrix(quantmatrix, zigZagList)
print(f'The flattened zig-zag matrix has length: {len(zigZagList)}\n')
# Then this zigZagList should be encoded to bytestream for even better compression
"""

idctmatrix = np.empty((len(tiles_1),len(tiles_1[0]),len(tiles_1[0][0]), len(tiles_1[0][0][0])), dtype='int16')
idctmatrix = dct3(quantmatrix)
print(f'biggest value of idctmatrix: {np.amax(idctmatrix)}\n')
print(f'The idctmatrix: \n {idctmatrix[50][50]}\n')

"""
requantmatrix = [[0 for x in range(len(tiles_1))]for x in range(len(tiles_1[0]))]
reQuantTheMatrix(idctmatrix, Q_50, requantmatrix)
print(f'The re-quantized matrix: \n {requantmatrix[50][50]}\n')
"""
newimg = np.empty((800,800), dtype='int16')
newimg = reshape_recompile(idctmatrix)
newimg += 128

show(newimg, "reshaped image")
