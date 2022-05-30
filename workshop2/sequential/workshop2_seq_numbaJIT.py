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
import csv

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

#@_time
@jit(nopython=True, parallel=False)    
def block_partition(image: np.ndarray, kernel_size: tuple, tester=0):
    block_per_dim = [int(len(image)/kernel_size[0]),int(len(image[0])/kernel_size[1])]
    tiled_array= np.empty((block_per_dim[0],block_per_dim[1],kernel_size[0],kernel_size[1]),dtype='int16')

    for x in range(block_per_dim[0]):
        for y in range(block_per_dim[1]):
            for i in range(kernel_size[0]):
                for j in range(kernel_size[1]):
                    tiled_array[x][y][i][j] = image[(kernel_size[0]*x)+i][(kernel_size[1]*y)+j]
    return tiled_array

#@_time
@jit(nopython=True, parallel=False)    
def recompile_image(image: np.ndarray, kernel_size: tuple, tester=0):
    block_per_dim = [len(image),len(image[0])]
    newimage = np.empty((block_per_dim[0]*kernel_size[0],block_per_dim[1]*kernel_size[1]),dtype='int16')
    #print(newimage.shape)
    for j in range(kernel_size[0]):
        for i in range(kernel_size[1]):
            for x in range(block_per_dim[1]): 
                for y in range(block_per_dim[0]):
                    newimage[(x*kernel_size[0])+i][(y*kernel_size
                    [1])+j] = image[x][y][i][j]
    return newimage
#@_time
def dct2_test(block):
    return dct(dct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')

#@_time
def dct3_test(block):
    return dct(dct(block.T, type=3, norm='ortho').T, type=3, norm='ortho')
#@_time
@jit(nopython=True, parallel=False)    
def dct2(image: np.ndarray, kernel_size: tuple, tester=0):                    # Array[P][Q][M][N]
    block_per_dim = [int(len(image)),int(len(image[0]))]
    dct_array = np.empty((block_per_dim[0],block_per_dim[1],kernel_size[0],kernel_size[1]),dtype='int16')
    dct_array = dct_array.astype('int16')
    block_sum = 0
    for x in range(block_per_dim[0]):
        #update_progress(x/block_per_dim[0])
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

@jit(nopython=True, parallel=False)    
def rescaleQuant(q50, scale):
    outMatrix = np.empty((len(Q_50),len(Q_50[0])), dtype='int16')
    outMatrix.astype('int16')
    for x in range(len(q50)):
        for y in range(len(q50[0])):
            outMatrix[x][y] = math.floor((scale*q50[x][y] + 50) / 100);
    return outMatrix

#@_time
@jit(nopython=True, parallel=False)    
def quantTheMatrix(matrix, q_s):
    outMatrix = np.empty((len(matrix),len(matrix[0]), len(matrix[0][0]), len(matrix[0][0][0])), dtype='int16')
    outMatrix.astype('int16')
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            for i in range(len(matrix[0][0])):
                outMatrix[x][y] = (matrix[x][y]/q_s)
    return outMatrix

#@_time
@jit(nopython=True, parallel=False)    
def idct2(image: np.ndarray, kernel_size: tuple, tester=0):                    # Array[P][Q][M][N]
    block_per_dim = [int(len(image)),int(len(image[0]))]
    idct_array = np.empty((block_per_dim[0],block_per_dim[1],kernel_size[0],kernel_size[1]),dtype='int16')    
    idct_array = idct_array.astype('int16')
    block_sum = 0
    for x in range(block_per_dim[0]):
        #update_progress(x/block_per_dim[0])
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

@jit(nopython=True, parallel=False)    
def reQuantTheMatrix(matrix, q_s):
    outMatrix = np.empty((len(matrix),len(matrix[0]), len(matrix[0][0]), len(matrix[0][0][0])), dtype='int16')
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            outMatrix[x][y] = (matrix[x][y]*q_s)
    return outMatrix
if __name__=="__main__": 
    
    sum_progress = 0
    total = 2*7*10
    result = np.empty((2,7,10))
    for x in range(2):
        if (x == 0):
            img = np.asarray(Image.open("Faur2.png").convert('L'), dtype='int16')
        else:
            img = np.asarray(Image.open("1600.jpg").convert('L'), dtype='int16')
        
        for y in range(2,9):
            n_cores = y 

            for z in range(10):
                start_final = time()
                img -= 128

                tilesize = (8,8)
                Q = 80

                # Tiling
                tiles_seq = block_partition(img, tilesize)

                
                # DCT2
                dct_seq = dct2(tiles_seq, tilesize, 1)


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
                
                Q_S = rescaleQuant(Q_50, S)


                # Quantization
                quant_seq = quantTheMatrix(dct_seq, Q_S)


                # Reconstruction
                requant_seq = reQuantTheMatrix(quant_seq, Q_S)


                # DCT2
                idct_seq = idct2(requant_seq, tilesize, 1)


                end_final = time()

                """
                #img_A = recompile_image(tiles, tilesize)
                img_A = recompile_image(tiles, tilesize)
                img_B = recompile_image(idct, tilesize)
                img_A += 128
                img_B += 128 
                plot_image = np.concatenate((img_A, img_B), axis=1)

                show(plot_image, "before and after")

                img_A = tiles[int(len(tiles)/2)][int(len(tiles)/2)]
                img_B = idct[int(len(idct)/2)][int(len(idct)/2)]
                img_A += 128
                img_B += 128 
                plot_image = np.concatenate((img_A, img_B), axis=1)
                show(plot_image, "before and after")
                """
                result[x][y-2][z] = end_final-start_final
                #assert np.allclose(tiles, idct,10), "The sequential and multi-processing  array's are not identical"
                sum_progress += 1
                update_progress(sum_progress/total)

    #assert np.allclose(tiles_seq, idct_seq,10), "The sequential and multi-processing  array's are not identical"

    with open('parallel_800.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = []
        for x in range(len(result[0])):
            header.append(f'{x+2} cores')
        
        # write the header
        writer.writerow(header)

        data = []
        for x in range(10):
            data.append([result[0][0][x], result[0][1][x], result[0][2][x], result[0][3][x], result[0][4][x], result[0][5][x], result[0][6][x]])
        
        # write multiple rows
        writer.writerows(data)

    with open('parallel_1600.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = []
        for x in range(len(result[0])):
            header.append(f'{x+2} cores')
        
        # write the header
        writer.writerow(header)

        data = []
        for x in range(10):
            data.append([result[1][0][x], result[1][1][x], result[1][2][x], result[1][3][x], result[1][4][x], result[1][5][x],result[1][6][x]])
        
        # write multiple rows
        writer.writerows(data)

