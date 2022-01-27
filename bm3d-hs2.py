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
import tqdm

from tqdm import trange
# ==================================================================================================
#                                           Preprocessing
# ==================================================================================================

def Initialization(Img):#blocksize 8  ,kaiser_window-beta 2.0
    InitImg = np.zeros(Img.shape, dtype=float)
    return InitImg

def dct2D(A):

    return dct(dct(A, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2D(A):
    """
    inverse 2D discrete cosine transform
    """

    return idct(idct(A, axis=0, norm='ortho'), axis=1, norm='ortho')


def PreDCT(Img, BlockSize):
    BlockDCT_all = np.zeros((Img.shape[0] - BlockSize, Img.shape[1] - BlockSize, BlockSize, BlockSize), \
                            dtype=float)

    for i in range(BlockDCT_all.shape[0]):
        for j in range(BlockDCT_all.shape[1]):
            Block = Img[i:i + BlockSize, j:j + BlockSize]
            BlockDCT_all[i, j, :, :] = dct2D(Block.astype(np.float64))

    return BlockDCT_all


# ==================================================================================================
#                                         Basic estimate
# ==================================================================================================

def Step1_Grouping(RefPoint, BlockDCT_all, BlockSize):

    BlockGroup = np.zeros((1, BlockSize, BlockSize), dtype=float) #(1024,8,8)
    RefDCT = BlockDCT_all[RefPoint[0], RefPoint[1], :, :]#(二维噪声DCT，8，8)从ref点开始的DCT  （8，8）
    BlockGroup[0, :, :] = RefDCT

    return BlockGroup

def Step1_3DFiltering(BlockGroup):
    ThreValue = lamb3d * sigma

    for i in range(BlockGroup.shape[1]):
        for j in range(BlockGroup.shape[2]):
            ThirdVector = dct(BlockGroup[:, i, j], norm='ortho')
            ThirdVector[abs(ThirdVector[:]) < ThreValue] = 0.
            BlockGroup[:, i, j] = list(idct(ThirdVector, norm='ortho'))
    return BlockGroup

def BM3D_Step1(noisyImg):
    BlockSize = Step1_BlockSize
    basicImg = Initialization(noisyImg)
    BlockDCT_all = PreDCT(noisyImg, BlockSize)

    for i in trange(noisyImg.shape[0]):
        for j in range(noisyImg.shape[1]):
            x = min(i, noisyImg.shape[0] - BlockSize - 1)
            y = min(j, noisyImg.shape[1] - BlockSize - 1)
            RefPoint = [x,y]
            BlockGroup = Step1_Grouping(RefPoint, BlockDCT_all, BlockSize)
            BlockGroup = Step1_3DFiltering(BlockGroup)
            temp = idct2D(np.squeeze(BlockGroup))
            #temp2 = idct2D(temp)
            basicImg[x:x+BlockSize, y:y+BlockSize] = temp

    return basicImg

# ==================================================================================================
#                                                main
# ==================================================================================================

if __name__ == '__main__':

    cv2.setUseOptimized(True)
    Step1_BlockSize = 3


    # ===============================================================================================
    # ============================================ BM3D =============================================
    path = r"D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\guidejbil\guide"
    savepath = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\hardshrinkage'
    imagename = 'DSC00039-01-'
    savename = r'hs-04-'

    print(os.path.join(savepath,imagename+'bm3d-'+savename+'r.npy'))
    print(os.path.join(savepath,imagename+'bm3d-'+savename+'r.raw'))
    out = np.load(os.path.join(path,imagename+'r.npy'))
    img = out.astype(np.float64)
    sigma = 70  # variance of the noise
    lamb3d = ((1.96*sigma)/(sigma*sigma))*sigma
    starttime = time.time()
    #步骤一
    basic_img = BM3D_Step1(img)
    endtime = time.time()
    print("time cost", starttime-endtime)
    basic_img_save = basic_img.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+'bm3d-'+savename+'r.npy'), basic_img_save)
    basic_img_save.tofile(os.path.join(savepath,imagename+'bm3d-'+savename+'r.raw'))

    print(os.path.join(savepath,imagename+'bm3d-'+savename+'b.npy'))
    print(os.path.join(savepath,imagename+'bm3d-'+savename+'b.raw'))
    out = np.load(os.path.join(path, imagename + 'b.npy'))
    img = out.astype(np.float64)
    sigma = 150
    lamb3d = ((1.96*sigma)/(sigma*sigma))*sigma
    starttime = time.time()
    #步骤一
    basic_img = BM3D_Step1(img)
    endtime = time.time()
    print("time cost", starttime-endtime)
    basic_img_save = basic_img.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+'bm3d-'+savename+'b.npy'), basic_img_save)
    basic_img_save.tofile(os.path.join(savepath,imagename+'bm3d-'+savename+'b.raw'))

    print(os.path.join(savepath,imagename+'bm3d-'+savename+'g1.npy'))
    print(os.path.join(savepath,imagename+'bm3d-'+savename+'g1.raw'))
    out = np.load(os.path.join(path, imagename + 'g1.npy'))
    img = out.astype(np.float64)
    sigma = 200
    lamb3d = ((1.96*sigma)/(sigma*sigma))*sigma
    starttime = time.time()
    #步骤一
    basic_img = BM3D_Step1(img)
    endtime = time.time()
    print("time cost", starttime-endtime)
    basic_img_save = basic_img.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+'bm3d-'+savename+'g1.npy'), basic_img_save)
    basic_img_save.tofile(os.path.join(savepath,imagename+'bm3d-'+savename+'g1.raw'))

    print(os.path.join(savepath,imagename+'bm3d-'+savename+'g2.npy'))
    print(os.path.join(savepath,imagename+'bm3d-'+savename+'g2.raw'))
    out = np.load(os.path.join(path, imagename + 'g2.npy'))
    img = out.astype(np.float64)
    sigma = 200
    lamb3d = ((1.96*sigma)/(sigma*sigma))*sigma
    starttime = time.time()
    #步骤一
    basic_img = BM3D_Step1(img)
    endtime = time.time()
    print("time cost", starttime-endtime)
    basic_img_save = basic_img.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+'bm3d-'+savename+'g2.npy'), basic_img_save)
    basic_img_save.tofile(os.path.join(savepath,imagename+'bm3d-'+savename+'g2.raw'))
