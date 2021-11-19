# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:28:17 2019
@author: Amos
Reference:
    [1] Image denoising by sparse 3D transform-domain collaborative filtering
    [2] An Analysis and Implementation of the BM3D Image Denoising Method

"""

import os
import cv2
import time
import sys
from scipy.fftpack import dct, idct
import numpy as np
from matplotlib import pylab as plb


# ==================================================================================================
#                                           Preprocessing
# ==================================================================================================

def AddNoise(Img, sigma):
    """
    Add Gaussian nosie to an image

    Return:
        nosiy image
    """

    GuassNoise = np.random.normal(0, sigma, Img.shape)

    noisyImg = Img + GuassNoise  # float type noisy image

    #    cv2.normalize(noisyImg, noisyImg, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    #
    #    noisyImg = noisyImg.astype(np.uint8)
    #
    #    cv2.imwrite('noisydog.png', noisyImg)
    #
    #    if cv2.imwrite('noisydog.png', noisyImg) == True:
    #
    #        print('Noise has been added to the original image.\n')
    #
    #        return noisyImg
    #
    #    else:
    #
    #        print('Error: adding noise failed.\n')
    #
    #        exit()

    return noisyImg


def Initialization(Img, BlockSize, Kaiser_Window_beta):#blocksize 8  ,kaiser_window-beta 2.0
    """
    Initialize the image, weight and Kaiser window

    Return:
        InitImg & InitWeight: zero-value Img.shape matrices
                  InitKaiser: (BlockSize * BlockSize) Kaiser window
    """
    InitImg = np.zeros(Img.shape, dtype=float)
    InitWeight = np.zeros(Img.shape, dtype=float)
    Window = np.matrix(np.kaiser(BlockSize, Kaiser_Window_beta))#（1，8）
    InitKaiser = np.array(Window.T * Window)#(8,8)

    return InitImg, InitWeight, InitKaiser


def SearchWindow(Img, RefPoint, BlockSize, WindowSize):
    """
    Find the search window whose center is reference block in *Img*
    Note that the center of SearchWindow is not always the reference block because of the border
    Return:
        (2 * 2) array of left-top and right-bottom coordinates in search window
    """
    if BlockSize >= WindowSize:
        print('Error: BlockSize is smaller than WindowSize.\n')
        exit()

    Margin = np.zeros((2, 2), dtype=int)
    Margin[0, 0] = max(0, RefPoint[0] + int((BlockSize - WindowSize) / 2))  # left-top x
    Margin[0, 1] = max(0, RefPoint[1] + int((BlockSize - WindowSize) / 2))  # left-top y
    Margin[1, 0] = Margin[0, 0] + WindowSize  # right-bottom x
    Margin[1, 1] = Margin[0, 1] + WindowSize  # right-bottom y
    if Margin[1, 0] >= Img.shape[0]:
        Margin[1, 0] = Img.shape[0] - 1
        Margin[0, 0] = Margin[1, 0] - WindowSize
    if Margin[1, 1] >= Img.shape[1]:
        Margin[1, 1] = Img.shape[1] - 1
        Margin[0, 1] = Margin[1, 1] - WindowSize

    return Margin

def dct2D(A):
    """
    2D discrete cosine transform (DCT)
    """

    return dct(dct(A, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2D(A):
    """
    inverse 2D discrete cosine transform
    """

    return idct(idct(A, axis=0, norm='ortho'), axis=1, norm='ortho')


def PreDCT(Img, BlockSize):
    """
    Do discrete cosine transform (2D transform) for each block in *Img* to reduce the complexity of
    applying transforms
    Return:
        BlockDCT_all: 4-dimensional array whose first two dimensions correspond to the block's
                      position and last two correspond to the DCT array of the block
    """

    BlockDCT_all = np.zeros((Img.shape[0] - BlockSize, Img.shape[1] - BlockSize, BlockSize, BlockSize), \
                            dtype=float)

    for i in range(BlockDCT_all.shape[0]):
        for j in range(BlockDCT_all.shape[1]):
            Block = Img[i:i + BlockSize, j:j + BlockSize]
            BlockDCT_all[i, j, :, :] = dct2D(Block.astype(np.float64))

    return BlockDCT_all


def ComputePSNR(Img1, Img2):
    """
    Compute the Peak Signal to Noise Ratio (PSNR) in decibles(dB).
    """

    if Img1.size != Img2.size:
        print('ERROR: two images should be in same size in computing PSNR.\n')

        sys.exit()

    Img1 = Img1.astype(np.float64)

    Img2 = Img2.astype(np.float64)

    RMSE = np.sqrt(np.sum((Img1 - Img2) ** 2) / Img1.size)

    return 20 * np.log10(18000. / RMSE)


# ==================================================================================================
#                                         Basic estimate
# ==================================================================================================

def Step1_Grouping(noisyImg, RefPoint, BlockDCT_all, BlockSize, ThreDist, MaxMatch, WindowSize):
    """
    Find blocks similar to the reference one in *noisyImg* based on *BlockDCT_all*

    Note that the distance computing is chosen from original paper rather than the analysis one
    Return:
          BlockPos: array of blocks' position (left-top point)
        BlockGroup: 3-dimensional array whose last two dimensions correspond to the DCT array of
                     the block
    """

    # initialization

    WindowLoc = SearchWindow(noisyImg, RefPoint, BlockSize, WindowSize)#一个矩阵（-15.5,-15.5,39,39)
    Block_Num_Searched = (WindowSize - BlockSize + 1) ** 2  # number of searched blocks  (32)2
    BlockPos = np.zeros((Block_Num_Searched, 2), dtype=int)#(1024,2)
    BlockGroup = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float) #(1024,8,8)
    Dist = np.zeros(Block_Num_Searched, dtype=float)#(32)2,一维数组 1024
    RefDCT = BlockDCT_all[RefPoint[0], RefPoint[1], :, :]#(二维噪声DCT，8，8)从ref点开始的DCT  （8，8）

    match_cnt = 0

    # Block searching and similarity (distance) computing

    for i in range(WindowSize - BlockSize + 1):#32
        for j in range(WindowSize - BlockSize + 1):

            SearchedDCT = BlockDCT_all[WindowLoc[0, 0] + i, WindowLoc[0, 1] + j, :, :]#滑动窗口 （8，8）
            dist = Step1_ComputeDist(RefDCT, SearchedDCT)

            if dist < ThreDist:
                BlockPos[match_cnt, :] = [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j]
                BlockGroup[match_cnt, :, :] = SearchedDCT
                Dist[match_cnt] = dist

                match_cnt += 1

    #    if match_cnt == 1:
    #
    #        print('WARNING: no similar blocks founded for the reference block {} in basic estimate.\n'\
    #              .format(RefPoint))

    if match_cnt <= MaxMatch:
        # less than MaxMatch similar blocks founded, return similar blocks
        BlockPos = BlockPos[:match_cnt, :]
        BlockGroup = BlockGroup[:match_cnt, :, :]
    else:
        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)  # indices of MaxMatch smallest distances
        BlockPos = BlockPos[idx[:MaxMatch], :]
        BlockGroup = BlockGroup[idx[:MaxMatch], :]
    #print(match_cnt)
    return BlockPos, BlockGroup


def Step1_Groupingmy(noisyImg, RefPoint, BlockDCT_all, BlockSize, ThreDist, MaxMatch, WindowSize,WindowSize2,Rate):
    """
    Find blocks similar to the reference one in *noisyImg* based on *BlockDCT_all*

    Note that the distance computing is chosen from original paper rather than the analysis one
    Return:
          BlockPos: array of blocks' position (left-top point)
        BlockGroup: 3-dimensional array whose last two dimensions correspond to the DCT array of
                     the block
    """

    # initialization

    WindowLoc = SearchWindow(noisyImg, RefPoint, BlockSize, WindowSize)  # 一个矩阵（-15.5,-15.5,39,39)
    Block_Num_Searched = (WindowSize - BlockSize + 1) ** 3  # number of searched blocks  (32)2
    BlockPos = np.zeros((Block_Num_Searched, 2), dtype=int)  # (1024,2)
    BlockGroup = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float)  # (1024,8,8)
    Dist = np.zeros(Block_Num_Searched, dtype=float)  # (32)2,一维数组 1024
    RefDCT = BlockDCT_all[RefPoint[0], RefPoint[1], :, :]  # (二维噪声DCT，8，8)从ref点开始的DCT  （8，8）

    match_cnt = 0

    # Block searching and similarity (distance) computing
    BlockPos[match_cnt, :] = RefPoint
    BlockGroup[match_cnt, :, :] = RefDCT
    Dist[match_cnt] = 0
    match_cnt += 1
    for i in range(0,WindowSize - BlockSize + 1,Rate):  # 32
        for j in range(0,WindowSize - BlockSize + 1,Rate):
            #print(i,j)
            SearchedDCT = BlockDCT_all[WindowLoc[0, 0] + i, WindowLoc[0, 1] + j, :, :]  # 滑动窗口 （8，8）
            dist = Step1_ComputeDist(RefDCT, SearchedDCT)
            #print(dist)
            if dist < ThreDist:
                BlockPos[match_cnt, :] = [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j]
                BlockGroup[match_cnt, :, :] = SearchedDCT
                Dist[match_cnt] = dist
                match_cnt += 1
                WindowLoc2 = SearchWindow(noisyImg, [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j], BlockSize, WindowSize2)
                for ii in range(1,WindowSize2):
                    for jj in range(1,WindowSize2):
                        if((WindowLoc2[0, 0] + ii)<(imagesize-BlockSize))&((WindowLoc2[0, 1] + jj)<(imagesize-BlockSize)):#图像size-blocksize
                            SearchedDCT2 = BlockDCT_all[WindowLoc2[0, 0] + ii, WindowLoc2[0, 1] + jj, :, :]
                            dist2 = Step1_ComputeDist(RefDCT, SearchedDCT2)
                            if dist2 < ThreDist:
                                BlockPos[match_cnt, :] = [WindowLoc2[0, 0] + ii, WindowLoc2[0, 1] + jj]

                                BlockGroup[match_cnt, :, :] = SearchedDCT2
                                Dist[match_cnt] = dist2
                                #print(Dist[match_cnt])
                                match_cnt += 1

    if match_cnt <= MaxMatch:
        # less than MaxMatch similar blocks founded, return similar blocks
        BlockPos = BlockPos[:match_cnt, :]
        BlockGroup = BlockGroup[:match_cnt, :, :]
    else:
        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)  # indices of MaxMatch smallest distances
        BlockPos = BlockPos[idx[:MaxMatch], :]
        BlockGroup = BlockGroup[idx[:MaxMatch], :]
    #print(match_cnt)
    return BlockPos, BlockGroup

def Step1_ComputeDist(BlockDCT1, BlockDCT2):
    """
    Compute the distance of two DCT arrays *BlockDCT1* and *BlockDCT2*
    """
    if BlockDCT1.shape != BlockDCT1.shape:
        print('ERROR: two DCT Blocks are not at the same shape in step1 computing distance.\n')
        sys.exit()

    elif BlockDCT1.shape[0] != BlockDCT1.shape[1]:
        print('ERROR: DCT Block is not square in step1 computing distance.\n')
        sys.exit()

    BlockSize = BlockDCT1.shape[0]

    if sigmaopen == True:
        ThreValue = lamb2d * sigma  #2,25
        BlockDCT1 = np.where(abs(BlockDCT1) < ThreValue, 0, BlockDCT1)
        BlockDCT2 = np.where(abs(BlockDCT2) < ThreValue, 0, BlockDCT2)
    #print(np.linalg.norm(BlockDCT1 - BlockDCT2) ** 2 / (BlockSize ** 2))
    return np.linalg.norm(BlockDCT1 - BlockDCT2) ** 2 / (BlockSize ** 2)#整体元素二范数，单个数值


def Step1_3DFiltering(BlockGroup):
    """
    Do collaborative hard-thresholding which includes 3D transform, noise attenuation through
    hard-thresholding and inverse 3D transform

    Return:
        BlockGroup
    """
    ThreValue = lamb3d * sigma    #2.7 5
    nonzero_cnt = 0

    # since 2D transform has been done, we do 1D transform, hard-thresholding and inverse 1D
    # transform, the inverse 2D transform is left in aggregation processing

    for i in range(BlockGroup.shape[1]):
        for j in range(BlockGroup.shape[2]):
            ThirdVector = dct(BlockGroup[:, i, j], norm='ortho')
            ThirdVector[abs(ThirdVector[:]) < ThreValue] = 0.
            # temp = np.nonzero(ThirdVector)[0].size
            # if (temp)>(Step1_MaxMatch-1):
            #     ThirdVector[ThirdVector[:] == ThirdVector.max()] = 0.



            # ThirdVector1mean = np.mean(ThirdVector)
            # ThirdVector[abs(ThirdVector[:])-ThirdVector1mean <= sigma] = 0.


            nonzero_cnt += np.nonzero(ThirdVector)[0].size
            BlockGroup[:, i, j] = list(idct(ThirdVector, norm='ortho'))#list()元组转换成列表

    return BlockGroup, nonzero_cnt


def Step1_Aggregation(BlockGroup, BlockPos, basicImg, basicWeight, basicKaiser, nonzero_cnt):
    """
    Compute the basic estimate of the true-image by weighted averaging all of the obtained
    block-wise estimates that are overlapping

    Note that the weight is set accroding to the original paper rather than the BM3D analysis one
    """

    if nonzero_cnt < 1:
        BlockWeight = 1.0 * basicKaiser
    else:
        BlockWeight = (1. / (sigma ** 2 * nonzero_cnt)) * basicKaiser

    for i in range(BlockPos.shape[0]):
        basicImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1], BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup.shape[2]] += BlockWeight * idct2D(BlockGroup[i, :, :])
        basicWeight[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1], BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup.shape[2]] += BlockWeight


def BM3D_Step1(noisyImg):
    """
    Give the basic estimate after grouping, collaborative filtering and aggregation
    Return:
        basic estimate basicImg
    """
    # preprocessing
    BlockSize = Step1_BlockSize
    ThreDist = Step1_ThreDist
    MaxMatch = Step1_MaxMatch
    WindowSize = Step1_WindowSize
    WindowSize2 = Step1_WindowSize2
    Rate = Step1_Rate
    spdup_factor = Step1_spdup_factor
    basicImg, basicWeight, basicKaiser = Initialization(noisyImg, BlockSize, Kaiser_Window_beta)
    BlockDCT_all = PreDCT(noisyImg, BlockSize)#(二维噪声DCT，8，8)

    # block-wise estimate with speed-up factor
    for i in range(int((noisyImg.shape[0] - BlockSize) / spdup_factor) + 2):
        for j in range(int((noisyImg.shape[1] - BlockSize) / spdup_factor) + 2):#  refpoint x
            RefPoint = [min(spdup_factor * i, noisyImg.shape[0] - BlockSize - 1), \
                        min(spdup_factor * j, noisyImg.shape[1] - BlockSize - 1)]
            if (i>10)&(j>10):
                BlockPos, BlockGroup = Step1_Groupingmy(noisyImg, RefPoint, BlockDCT_all, BlockSize, \
                                                      ThreDist, MaxMatch, WindowSize,WindowSize2,Rate)
            else:
                BlockPos, BlockGroup = Step1_Grouping(noisyImg, RefPoint, BlockDCT_all, BlockSize, \
                                                        ThreDist, MaxMatch, WindowSize)
            BlockGroup, nonzero_cnt = Step1_3DFiltering(BlockGroup)
            Step1_Aggregation(BlockGroup, BlockPos, basicImg, basicWeight, basicKaiser, nonzero_cnt)

    basicWeight = np.where(basicWeight == 0, 1, basicWeight)
    basicImg[:, :] /= basicWeight[:, :]
    #    basicImg = (np.matrix(basicImg, dtype=int)).astype(np.uint8)

    return basicImg


# ==================================================================================================
#                                         Final estimate
# ==================================================================================================

def Step2_Grouping(basicImg, noisyImg, RefPoint, BlockSize, ThreDist, MaxMatch, WindowSize,
                   BlockDCT_basic, BlockDCT_noisy):
    """
    Similar to Step1_Grouping, find the similar blocks to the reference one from *basicImg*

    Return:
                BlockPos: array of similar blocks' position (left-top point)
        BlockGroup_basic: 3-dimensional array standing for the stacked blocks similar to the
                          reference one from *basicImg* after 2D DCT
        BlockGroup_noisy: the stacked blocks from *noisyImg* corresponding to BlockGroup_basic
    """

    # initialization (same as Step1)

    WindowLoc = SearchWindow(basicImg, RefPoint, BlockSize, WindowSize)#一个矩阵（-15.5,-15.5,39,39)搜索框
    Block_Num_Searched = (WindowSize - BlockSize + 1) ** 2# number of searched blocks  (32)2 匹配块矩阵个数预设置
    BlockPos = np.zeros((Block_Num_Searched, 2), dtype=int)#(1024,2) 匹配块个数矩阵设置
    BlockGroup_basic = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float) #(1024,8,8)
    BlockGroup_noisy = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float)
    Dist = np.zeros(Block_Num_Searched, dtype=float)#(32)2,一维数组 1024 真实匹配到的个数限制

    match_cnt = 0

    # Block searching and similarity (distance) computing
    # Note the distance computing method is different from that of Step1

    for i in range(WindowSize - BlockSize + 1):
        for j in range(WindowSize - BlockSize + 1):
            SearchedPoint = [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j]
            dist = Step2_ComputeDist(basicImg, RefPoint, SearchedPoint, BlockSize)
            if dist < ThreDist:
                BlockPos[match_cnt, :] = SearchedPoint
                Dist[match_cnt] = dist
                match_cnt += 1

    #print(match_cnt)
    if match_cnt <= MaxMatch:
        # less than MaxMatch similar blocks founded, return similar blocks
        BlockPos = BlockPos[:match_cnt, :]
    else:
        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)  # indices of MaxMatch smallest distances
        BlockPos = BlockPos[idx[:MaxMatch], :]
    for i in range(BlockPos.shape[0]):
        SimilarPoint = BlockPos[i, :]
        BlockGroup_basic[i, :, :] = BlockDCT_basic[SimilarPoint[0], SimilarPoint[1], :, :]
        BlockGroup_noisy[i, :, :] = BlockDCT_noisy[SimilarPoint[0], SimilarPoint[1], :, :]
    BlockGroup_basic = BlockGroup_basic[:BlockPos.shape[0], :, :]
    BlockGroup_noisy = BlockGroup_noisy[:BlockPos.shape[0], :, :]

    return BlockPos, BlockGroup_basic, BlockGroup_noisy
def Step2_Groupingmy(basicImg, noisyImg, RefPoint, BlockSize, ThreDist, MaxMatch, WindowSize,
                   BlockDCT_basic, BlockDCT_noisy,Rate,WindowSize2):

    # initialization (same as Step1)
    WindowLoc = SearchWindow(basicImg, RefPoint, BlockSize, WindowSize)#一个矩阵（-15.5,-15.5,39,39)搜索框
    Block_Num_Searched = (WindowSize - BlockSize + 1) ** 3# number of searched blocks  (32)2 匹配块矩阵个数预设置
    BlockPos = np.zeros((Block_Num_Searched, 2), dtype=int)#(1024,2) 匹配块个数矩阵设置
    BlockGroup_basic = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float) #(1024,8,8)
    BlockGroup_noisy = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float)
    Dist = np.zeros(Block_Num_Searched, dtype=float)#(32)2,一维数组 1024 真实匹配到的个数限制

    match_cnt = 0
    BlockPos[match_cnt, :] = RefPoint
    Dist[match_cnt] = 0
    match_cnt += 1
    for i in range(0,WindowSize - BlockSize + 1,Rate):
        for j in range(0,WindowSize - BlockSize + 1,Rate):
            SearchedPoint = [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j]
            dist = Step2_ComputeDist(basicImg, RefPoint, SearchedPoint, BlockSize)
            if dist < ThreDist:
                BlockPos[match_cnt, :] = SearchedPoint
                Dist[match_cnt] = dist
                match_cnt += 1
                WindowLoc2 = SearchWindow(noisyImg, [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j], BlockSize, WindowSize2)
                for ii in range(1,WindowSize2):
                    for jj in range(1,WindowSize2):
                        if((WindowLoc2[0, 0] + ii)<(imagesize-BlockSize))&((WindowLoc2[0, 1] + jj)<(imagesize-BlockSize)):#图像size-blocksize
                            SearchedPoint2 = [WindowLoc2[0, 0] + ii, WindowLoc2[0, 1] + jj]
                            dist2 = Step2_ComputeDist(basicImg, RefPoint, SearchedPoint2, BlockSize)
                            if dist2 < ThreDist:
                                BlockPos[match_cnt, :] = SearchedPoint2
                                Dist[match_cnt] = dist2
                                match_cnt += 1
    #print(match_cnt)
    if match_cnt <= MaxMatch:
        # less than MaxMatch similar blocks founded, return similar blocks
        BlockPos = BlockPos[:match_cnt, :]
    else:
        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)  # indices of MaxMatch smallest distances
        BlockPos = BlockPos[idx[:MaxMatch], :]
    for i in range(BlockPos.shape[0]):
        SimilarPoint = BlockPos[i, :]
        BlockGroup_basic[i, :, :] = BlockDCT_basic[SimilarPoint[0], SimilarPoint[1], :, :]
        BlockGroup_noisy[i, :, :] = BlockDCT_noisy[SimilarPoint[0], SimilarPoint[1], :, :]
    BlockGroup_basic = BlockGroup_basic[:BlockPos.shape[0], :, :]
    BlockGroup_noisy = BlockGroup_noisy[:BlockPos.shape[0], :, :]

    return BlockPos, BlockGroup_basic, BlockGroup_noisy

def Step2_ComputeDist(img, Point1, Point2, BlockSize):
    """
    Compute distance between blocks whose left-top margins' coordinates are *Point1* and *Point2*
    """

    Block1 = (img[Point1[0]:Point1[0] + BlockSize, Point1[1]:Point1[1] + BlockSize]).astype(np.float64)

    Block2 = (img[Point2[0]:Point2[0] + BlockSize, Point2[1]:Point2[1] + BlockSize]).astype(np.float64)
    #print(np.linalg.norm(Block1 - Block2) ** 2 / (BlockSize ** 2))
    return np.linalg.norm(Block1 - Block2) ** 2 / (BlockSize ** 2)


def Step2_3DFiltering(BlockGroup_basic, BlockGroup_noisy):
    """
    Do collaborative Wiener filtering and here we choose 2D DCT + 1D DCT as the 3D transform which
    is the same with the 3D transform in hard-thresholding filtering
    Note that the Wiener weight is set accroding to the BM3D analysis paper rather than the original
    one
    Return:
       BlockGroup_noisy & WienerWeight
    """
    #sigma = sigma2
    Weight = 0
    coef = 1.0 / BlockGroup_noisy.shape[0]

    for i in range(BlockGroup_noisy.shape[1]):
        for j in range(BlockGroup_noisy.shape[2]):
            Vec_basic = dct(BlockGroup_basic[:, i, j], norm='ortho')
            Vec_noisy = dct(BlockGroup_noisy[:, i, j], norm='ortho')

            Vec_value = Vec_basic**2*coef#方法一非原文
            #Vec_value = Vec_basic.T * Vec_basic#方法二
            Vec_value /= (Vec_value + sigma **2)  # pixel weight

            Vec_noisy *= Vec_value
            Weight += np.sum(Vec_value)
            #            for k in range(BlockGroup_noisy.shape[0]):
            #                Value = Vec_basic[k]**2 * coef
            #                Value /= (Value + sigma**2) # pixel weight
            #                Vec_noisy[k] = Vec_noisy[k] * Value
            #                Weight += Value
            BlockGroup_noisy[:, i, j] = list(idct(Vec_noisy, norm='ortho'))

    if Weight > 0:
        WienerWeight = 1. / (sigma ** 2 * (Weight))
    else:
        WienerWeight = 1.0

    return BlockGroup_noisy, WienerWeight

def Step2_Aggregation(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight, finalKaiser):
    """
    Compute the final estimate of the true-image by aggregating all of the obtained local estimates
    using a weighted average
    """

    BlockWeight = WienerWeight * finalKaiser

    for i in range(BlockPos.shape[0]):
        finalImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1],BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup_noisy.shape[2]] += BlockWeight * idct2D(BlockGroup_noisy[i, :, :])

        finalWeight[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup_noisy.shape[1],BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup_noisy.shape[2]] += BlockWeight


def BM3D_Step2(basicImg, noisyImg):
    """
    Give the final estimate after grouping, Wiener filtering and aggregation
    Return:
        final estimate finalImg
    """
    # parameters setting

    BlockSize = Step2_BlockSize
    ThreDist = Step2_ThreDist
    MaxMatch = Step2_MaxMatch
    WindowSize = Step2_WindowSize
    spdup_factor = Step2_spdup_factor
    Rate = Step2_Rate
    WindowSize2 = Step2_WindowSize2

    finalImg, finalWeight, finalKaiser = Initialization(basicImg, BlockSize, Kaiser_Window_beta)
    BlockDCT_noisy = PreDCT(noisyImg, BlockSize)#噪声图DCT
    BlockDCT_basic = PreDCT(basicImg, BlockSize)#S1DCT

    # block-wise estimate with speed-up factor

    for i in range(int((basicImg.shape[0] - BlockSize) / spdup_factor) + 2):
        for j in range(int((basicImg.shape[1] - BlockSize) / spdup_factor) + 2):#基准点，
            RefPoint = [min(spdup_factor * i, basicImg.shape[0] - BlockSize - 1), \
                        min(spdup_factor * j, basicImg.shape[1] - BlockSize - 1)]#以bsicimg进行gouping,创建
            if (i < 3) & (j < 3):
                BlockPos, BlockGroup_basic, BlockGroup_noisy = Step2_Grouping(basicImg, noisyImg, \
                                                                              RefPoint, BlockSize, \
                                                                              ThreDist, MaxMatch, \
                                                                              WindowSize, \
                                                                              BlockDCT_basic, \
                                                                              BlockDCT_noisy)
            else:
                BlockPos, BlockGroup_basic, BlockGroup_noisy = Step2_Groupingmy(basicImg, noisyImg, \
                                                                              RefPoint, BlockSize, \
                                                                              ThreDist, MaxMatch, \
                                                                              WindowSize, \
                                                                              BlockDCT_basic, \
                                                                              BlockDCT_noisy,Rate,WindowSize2)
            BlockGroup_noisy, WienerWeight = Step2_3DFiltering(BlockGroup_basic, BlockGroup_noisy)
            Step2_Aggregation(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight, \
                              finalKaiser)
    finalWeight = np.where(finalWeight == 0, 1, finalWeight)
    finalImg[:, :] /= finalWeight[:, :]

    return finalImg


# ==================================================================================================
#                                                main
# ==================================================================================================

if __name__ == '__main__':

    cv2.setUseOptimized(True)
    #加载图片

    path = r"D:\Project\Python\Data\Data\sources\bm3d\newgama\400\1917-R.npy"
    out = np.load(path)
    img = out.astype(np.float64)
    # ================================== Parameters initialization ==================================
    imagesize = 400
    sigma = 163  # variance of the noise
    #sigma2 = sigma   #相当于平滑程度
    sigmaopen = False
    lamb2d = 2*sigma#orgin2
    lamb3d = 0.012*sigma  #origin2.7  (2-3)   95%(+-1.96sigma 500)/sigma**2  0.0037     0.014 0.009 0.012
    Step1_ThreDist = 8000  # threshold distance
    Step1_MaxMatch = 30  # max matched blocks
    Step1_BlockSize = 5   #要大于spdup_factor
    Step1_spdup_factor = 3 # pixel jump for new reference block  (3-6)
    Step1_WindowSize = 30  # search window size origin39
    Step1_WindowSize2 = 6  #二次定位大小
    Step1_Rate = 6 #搜索框步长

    Step2_ThreDist = 300
    Step2_MaxMatch = 15
    Step2_BlockSize = 3
    Step2_spdup_factor = 3
    Step2_WindowSize = 15
    Step2_WindowSize2 = 4  #二次定位大小
    Step2_Rate = 4 #搜索框步长

    Kaiser_Window_beta = 2.0
    # ===============================================================================================

    # ============================================ BM3D =============================================

    starttime = time.time()

    #步骤一
    basic_img = BM3D_Step1(img)
    endtime = time.time()
    print("time cost", starttime-endtime)
    basic_img_save = basic_img.astype(np.uint16)



    #步骤二
    # basic_img = basic_img.astype(np.float64)
    # final_img = BM3D_Step2(basic_img, img)
    # endtime2 = time.time()
    # print("time cost", starttime-endtime2)
    # final = final_img.astype(np.uint16)

    #保存
    basic_img_save.tofile(r'D:\Project\Python\Data\Data\sources\bm3d\newgama\400\TURBO\1917-R-s1.npy')
    np.save(r'D:\Project\Python\Data\Data\sources\bm3d\newgama\400\TURBO\1917-R-s1.npy', basic_img_save)



    #计算PSNR
    # real = np.load(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-300.npy')
    # real = real.astype(np.uint16)
    # basic_PSNR = ComputePSNR(real, basic_img)
    # print('The PSNR of basic image is {} dB.\n'.format(basic_PSNR))
    # final_PSNR = ComputePSNR(real, final_img)
    # print('The PSNR of final image is {} dB.\n'.format(final_PSNR))

