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
import os, sys
from itertools import chain
import argparse
from mpi4py import MPI
import multiprocessing as mp
from numba import prange, jit, vectorize

def _time(f):
    def wrapper(*args):
        start = time()
        r = f(*args)                            
        end = time()
        print("%s timed %f" % (f.__name__, end-start) )
        return r
    return wrapper

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

def update_progress(progress):
        """
        Displays or updates a console progress bar
        Args
            progress (float): Accepts a float between 0 and 1. Any int will be converted to a float. (A value under 0 represents a 'halt'.) (A value at 1 or bigger represents 100%)
                        
        Returns:
            progression bar (terminal output)
        """
        barLength = 23 # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength*progress))
        text = "\rCreating correctionList: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), int(progress*100), status)
        sys.stdout.write(text)
        sys.stdout.flush()

@_time
def block_partition(image: np.ndarray, kernel_size: tuple, tester=0):
    imagef = image.copy().astype(np.float32)
    if (tester==0):
        print(image.shape)
        img_height, img_width = imagef.shape
        tile_height, tile_width = kernel_size
        tiled_array = image.reshape(img_height // tile_height,
                                    tile_height,
                                    img_width // tile_width,
                                    tile_width
                                    )
        tiled_array = tiled_array.swapaxes(1, 2)
        return tiled_array
    else:
        block_per_dim = [int(len(image)/kernel_size[0]),int(len(image[0])/kernel_size[1])]
        tiled_array= np.empty((block_per_dim[0],block_per_dim[1],kernel_size[0],kernel_size[1]),dtype='int16')

        for x in range(block_per_dim[0]):
            for y in range(block_per_dim[1]):
                for i in range(kernel_size[0]):
                    tiled_array[x][y][i] = [image.item((kernel_size[0]*x)+i,(kernel_size[1]*y)+j) for j in range(kernel_size[1])]
        return tiled_array

@_time
def recompile_image(image: np.ndarray, kernel_size: tuple, tester=0):
    if (tester==0):
        image = image.swapaxes(1, 2)
        newimage = image.reshape(-1, 800)

        return newimage
    else:
        image = image.swapaxes(1, 2)
        newimage = image.reshape(-1, 800)

    return newimage

#@_time
def dct2_test(block):
    return dct(dct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')

#@_time
def dct3_test(block):
    return dct(dct(block.T, type=3, norm='ortho').T, type=3, norm='ortho')
@_time
def dct2(image: np.ndarray, kernel_size: tuple, tester=0):                    # Array[P][Q][M][N]
    block_per_dim = [int(len(image)),int(len(image[0]))]
    dct_array = np.empty((block_per_dim[0],block_per_dim[1],kernel_size[0],kernel_size[1]),dtype='int16')
    print(f'block per dimension: {block_per_dim}')
    if (tester!=0):
        block_sum = 0
        for x in range(block_per_dim[0]):
            update_progress(x/block_per_dim[0])
            for y in range(block_per_dim[1]):
                for i in range(kernel_size[0]):
                    if (i == 0):
                        scalar_i =1/math.sqrt(kernel_size[0]) 
                    else:
                        scalar_i = math.sqrt(2/kernel_size[0])
                    for j in range(kernel_size[1]):
                        if (j == 0):
                            scalar_j = 1/math.sqrt(kernel_size[1])
                        else:
                            scalar_j = math.sqrt(2/kernel_size[1])
                        for m in range(kernel_size[0]):
                            for n in range(kernel_size[1]):
                                block_sum += image[x][y][m][n]*math.cos((math.pi*(2*m+1)*i)/(2*kernel_size[0]))*math.cos((math.pi*(2*n+1)*j)/(2*kernel_size[1]))

                        dct_array[x][y][i][j] = scalar_i*scalar_j*block_sum
                        block_sum = 0
        return dct_array
    else: # Skal laves om
        for x in range(block_per_dim[0]):
            update_progress(x/block_per_dim[0])
            for y in range(block_per_dim[1]):
                dct_array[x][y] = dct2_test(image[x][y])
        return dct_array

def rescaleQuant(q50, scale, outMatrix):
    for x in range(len(q50)):
        for y in range(len(q50[0])):
            outMatrix[x][y] = math.floor((scale*q50[x][y] + 50) / 100);
    return outMatrix

@_time
def quantTheMatrix(matrix, q_s):
    outMatrix = np.empty((len(matrix),len(matrix[0]), len(matrix[0][0]), len(matrix[0][0][0])), dtype='int16')
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            for i in range(len(matrix[0][0])):
                outMatrix[x][y] = np.round((matrix[x][y]/q_s), 0)
    return outMatrix

@_time
def idct2(image: np.ndarray, kernel_size: tuple, tester=0):                    # Array[P][Q][M][N]
    block_per_dim = [int(len(image)),int(len(image[0]))]
    idct_array = np.empty((block_per_dim[0],block_per_dim[1],kernel_size[0],kernel_size[1]),dtype='int16')
    print(f'block per dimension: {block_per_dim}')
    if (tester!=0):
        block_sum = 0
        for x in range(block_per_dim[0]):
            update_progress(x/block_per_dim[0])
            for y in range(block_per_dim[1]):
                for m in range(kernel_size[0]):
                    
                    for n in range(kernel_size[1]):
                        
                        for i in range(kernel_size[0]):
                            if (i == 0):
                                scalar_i =1/math.sqrt(kernel_size[0]) 
                            else:
                                scalar_i = math.sqrt(2/kernel_size[0])

                            for j in range(kernel_size[1]):
                                if (j == 0):
                                    scalar_j = 1/math.sqrt(kernel_size[1])
                                else:
                                    scalar_j = math.sqrt(2/kernel_size[1])
                                block_sum += scalar_i*scalar_j*image[x][y][i][j]*math.cos((math.pi*(2*m+1)*i)/(2*kernel_size[0]))*math.cos((math.pi*(2*n+1)*j)/(2*kernel_size[1]))

                        idct_array[x][y][m][n] = block_sum
                        block_sum = 0
        return idct_array
    else: # Skal laves om
        for x in range(block_per_dim[0]):
            update_progress(x/block_per_dim[0])
            for y in range(block_per_dim[1]):
                idct_array[x][y] = dct3_test(image[x][y])
        return idct_array

def reQuantTheMatrix(matrix, q_s):
    outMatrix = np.empty((len(matrix),len(matrix[0]), len(matrix[0][0]), len(matrix[0][0][0])), dtype='int16')
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            outMatrix[x][y] = np.round((matrix[x][y]*q_s), 0)
    return outMatrix
if __name__=="__main__": 
    

    # Arguments for testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_vector", action="store_true")
    parser.add_argument("--seq_numbaJIT", action="store_true")
    parser.add_argument("--seq_numbaVector", action="store_true")
    parser.add_argument("--mp", action="store_true")
    parser.add_argument("--mp_numbaJIT", action="store_true")
    parser.add_argument("--mp_GPU", action="store_true")
    parser.add_argument("--mp_MPI", action="store_true")
    args = parser.parse_args()

    """if args.space:
                    timeOrSpace = 1
                
                if args.print:
                    printOut = 1
            
                if args.list:
                    motorCorrection(ImProcOutput,cameraFOV,NumberOfSpeedSettings,0,timeOrSpace,printOut)
                else:
                    with open('correctionList.pkl', 'rb') as f:
                        correctionList = pickle.load(f)
                    motorCorrection(ImProcOutput,cameraFOV,NumberOfSpeedSettings,correctionList,timeOrSpace,printOut)"""
    
    # Testing
    #imageX = 80
    #imageY = 80
    #img = np.random.randint(0,255,(imageX,imageY), dtype='int16')

    img = np.asarray(Image.open("Faur2.png").convert('L'), dtype='int16')
    img -= 128

    tilesize = (8,8)
    Q = 80

    # Tiling
    vblock_partition = np.vectorize(block_partition)
    tiles_mp = vblock_partition(img, tilesize)
    print(tiles_mp[int(len(tiles_mp)/2)][int(len(tiles_mp)/2)])
    tiles_seq = vblock_partition(img, tilesize,1)
    print(tiles_seq[int(len(tiles_seq)/2)][int(len(tiles_seq)/2)])
    tiles_check = tiles_seq == tiles_mp
    assert tiles_check.all(), "The sequential and multi-processing  array's are not identical"

    # DCT2
    dct_mp = dct2(tiles_mp, tilesize)
    print(dct_mp[int(len(dct_mp)/2)][int(len(dct_mp)/2)])
    dct_seq = dct2(tiles_seq, tilesize, 1)
    print(dct_seq[int(len(dct_seq)/2)][int(len(dct_seq)/2)])

    #assert np.allclose(dct_seq, dct_mp,1), "The sequential and multi-processing  array's are not identical"

    """
    Source: https://www.math.cuhk.edu.hk/~lmlui/dct.pdf
    There exist a standardizezd quantization matrix where values range from 0-100, 100 is high quality and low compression and 0 is high compression and low quality
    Below is the 50 version
    """

    # Quantization
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
    print(Q_50)
    print(Q_S)

    # Quantization
    quant_seq = quantTheMatrix(dct_seq, Q_S)
    print(quant_seq[int(len(quant_seq)/2)][int(len(quant_seq)/2)])
    quant_mp = quantTheMatrix(dct_mp, Q_S)
    print(quant_mp[int(len(quant_mp)/2)][int(len(quant_mp)/2)])

    # Reconstruction
    requant_seq = reQuantTheMatrix(quant_seq, Q_S)
    print(requant_seq[int(len(requant_seq)/2)][int(len(requant_seq)/2)])
    requant_mp = reQuantTheMatrix(quant_mp, Q_S)
    print(requant_mp[int(len(requant_mp)/2)][int(len(requant_mp)/2)])

    # DCT2
    idct_mp = idct2(requant_mp, tilesize)
    print(idct_mp[int(len(idct_mp)/2)][int(len(idct_mp)/2)])
    idct_seq = idct2(requant_seq, tilesize, 1)
    print(idct_seq[int(len(idct_seq)/2)][int(len(idct_seq)/2)])


    img_A = recompile_image(tiles_seq, tilesize)
    img_B = recompile_image(idct_seq, tilesize)
    img_A += 128
    img_B += 128 
    plot_image = np.concatenate((img_A, img_B), axis=1)

    show(plot_image, "before and after")

    img_A = tiles_seq[int(len(tiles_seq)/2)][int(len(tiles_seq)/2)]
    img_B = idct_seq[int(len(idct_seq)/2)][int(len(idct_seq)/2)]
    img_A += 128
    img_B += 128 
    plot_image = np.concatenate((img_A, img_B), axis=1)
    show(plot_image, "before and after")

    #assert np.allclose(tiles_seq, idct_seq,10), "The sequential and multi-processing  array's are not identical"



