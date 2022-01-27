#联合双边滤波
#图像高斯平滑，近似性权重由高斯平滑后确定
import numpy as np
from scipy import signal
import cv2,math
import os
import time
from matplotlib import pylab as plb
import tqdm
from tqdm import trange

def getClosenessWeight(sigma_g,H,W):
    r,c = np.mgrid[0:H:1,0:W:1]
    r=r.astype(np.float64)
    c=c.astype(np.float64)
    r-=(H-1)/2
    c-=(W-1)/2
    closeWeight = np.exp(-0.5*(np.power(r,2)+np.power(c,2))/math.pow(sigma_g,2))
    return closeWeight

def jointBLFrbg(Ir,Ib,Ig1,Ig2,H,W,sigma_gr,sigma_dr,sigma_gb,sigma_db,sigma_gg1,sigma_dg1,sigma_gg2,sigma_dg2,borderType=cv2.BORDER_CONSTANT):
    closenessWeightr = getClosenessWeight(sigma_gr,H,W)#高斯距离权重
    closenessWeightb = getClosenessWeight(sigma_gb, H, W)
    closenessWeightg1 = getClosenessWeight(sigma_gg1, H, W)
    closenessWeightg2 = getClosenessWeight(sigma_gg2, H, W)
    cH = int((H-1)/2)#1
    cW = int((W-1)/2)

    I = Ir.astype(np.uint16)
    Igr = cv2.GaussianBlur(Ir,(W,H),sigma_gr)#高斯引导图r
    Igb = cv2.GaussianBlur(Ib, (W, H), sigma_gb)  # 高斯引导图b
    Igg1 = cv2.GaussianBlur(Ig1, (W, H), sigma_gg1)  # 高斯引导图g1
    Igg2 = cv2.GaussianBlur(Ig2, (W, H), sigma_gg2)  # 高斯引导图g2

    Ipr = cv2.copyMakeBorder(Ir,cH,cH,cW,cW,borderType,value = 0)#噪声图r
    Ipb = cv2.copyMakeBorder(Ib, cH, cH, cW, cW, borderType,value = 0)  # 噪声图b
    Ipg1 = cv2.copyMakeBorder(Ig1, cH, cH, cW, cW, borderType,value = 0)  # 噪声图g1
    Ipg2 = cv2.copyMakeBorder(Ig2, cH, cH, cW, cW, borderType,value = 0)  # 噪声图g2
    Igpr = cv2.copyMakeBorder(Igr,cH,cH,cW,cW,borderType,value = 0)#高斯滤波图r
    Igpb = cv2.copyMakeBorder(Igb, cH, cH, cW, cW, borderType,value = 0)  # 高斯滤波图b
    Igpg1 = cv2.copyMakeBorder(Igg1, cH, cH, cW, cW, borderType,value = 0)  # 高斯滤波图g1
    Igpg2 = cv2.copyMakeBorder(Igg2, cH, cH, cW, cW, borderType,value = 0)  # 高斯滤波图g2

    rows,cols = I.shape
    i,j = 0,0
    jblfr = np.zeros(I.shape,np.float64)
    jblfb = np.zeros(I.shape, np.float64)
    jblfg1 = np.zeros(I.shape, np.float64)
    jblfg2 = np.zeros(I.shape, np.float64)
    for r in trange(cH,cH+rows,1):
        for c in np.arange(cW,cW+cols,1):
            rTop,rBottom = r-cH,r+cH
            cLeft,cRight = c-cW,c+cW

            pixelr = Igpr[r][c]
            pixelb = Igpb[r][c]
            pixelg1 = Igpg1[r][c]
            pixelg2 = Igpg2[r][c]

            regionr = Igpr[rTop:rBottom+1,cLeft:cRight+1]
            regionb = Igpb[rTop:rBottom + 1, cLeft:cRight + 1]
            regiong1 = Igpg1[rTop:rBottom + 1, cLeft:cRight + 1]
            regiong2 = Igpg2[rTop:rBottom + 1, cLeft:cRight + 1]

            similarityWeightr = np.exp(-0.5*np.power(regionr-pixelr,2.0)/math.pow(sigma_dr,2.0))
            similarityWeightb = np.exp(-0.5 * np.power(regionb - pixelb, 2.0) / math.pow(sigma_db, 2.0))
            similarityWeightg1 = np.exp(-0.5 * np.power(regiong1 - pixelg1, 2.0) / math.pow(sigma_dg1, 2.0))
            similarityWeightg2 = np.exp(-0.5 * np.power(regiong2 - pixelg2, 2.0) / math.pow(sigma_dg2, 2.0))

            weightr = closenessWeightr*similarityWeightr*similarityWeightb*similarityWeightg1*similarityWeightg2
            weightb = closenessWeightb*similarityWeightb*similarityWeightg1*similarityWeightg2
            weightg1 = closenessWeightg1*similarityWeightg1*similarityWeightg2
            weightg2 = closenessWeightg2*similarityWeightg1*similarityWeightg2

            weightr = weightr/np.sum(weightr)
            weightb = weightb / np.sum(weightb)
            weightg1 = weightg1 / np.sum(weightg1)
            weightg2 = weightg2 / np.sum(weightg2)

            jblfr[i][j] = np.sum(Ipr[rTop:rBottom+1,cLeft:cRight+1]*weightr)
            jblfb[i][j] = np.sum(Ipb[rTop:rBottom + 1, cLeft:cRight + 1] * weightb)
            jblfg1[i][j] = np.sum(Ipg1[rTop:rBottom + 1, cLeft:cRight + 1] * weightg1)
            jblfg2[i][j] = np.sum(Ipg2[rTop:rBottom + 1, cLeft:cRight + 1] * weightg2)
            j+=1
        j = 0
        i+=1
    return jblfr,jblfb,jblfg1,jblfg2

if __name__ =='__main__':
    path = r'D:\Project\Python\Data\Data\sources\temp\6946\SFR\origin\14bit'
    savepath = r'D:\Project\Python\Data\Data\sources\temp\6946\SFR\jbil'
    imagename = 'lsb5-14bit-'
    newname = 'jbillsb-'
    num = '101520-'
    winsize = 3

    sigmadr = 10
    sigmarr = 10
    sigmadb = 15
    sigmarb = 15
    sigmadg1 = 1
    sigmarg1 = 20
    sigmadg2 = 1
    sigmarg2 = 20

    Ir = np.load(os.path.join(path,imagename+'r.npy'))
    fIr = Ir.astype(np.float64)
    Ib = np.load(os.path.join(path, imagename + 'b.npy'))
    fIb = Ib.astype(np.float64)
    Ig1 = np.load(os.path.join(path,imagename+'g1.npy'))
    fIg1 = Ig1.astype(np.float64)
    Ig2 = np.load(os.path.join(path,imagename+'g2.npy'))
    fIg2 = Ig2.astype(np.float64)
    jblfr,jblfb,jblfg1,jblfg2 = jointBLFrbg(fIr,fIb,fIg1,fIg2,winsize,winsize,sigmadr,sigmarr,sigmadb,sigmarb,sigmadg1,sigmarg1,sigmadg2,sigmarg2)
    jblfint16r = jblfr.astype(np.uint16)
    np.save(os.path.join(savepath,newname+num+'r.npy'),jblfint16r)
    jblfint16r.tofile(os.path.join(savepath,newname+num+'r.raw'))
    jblfint16b = jblfb.astype(np.uint16)
    np.save(os.path.join(savepath,newname+num+'b.npy'),jblfint16b)
    jblfint16b.tofile(os.path.join(savepath,newname+num+'b.raw'))
    jblfint16g1 = jblfg1.astype(np.uint16)
    np.save(os.path.join(savepath,newname+num+'g1.npy'),jblfint16g1)
    jblfint16g1.tofile(os.path.join(savepath,newname+num+'g1.raw'))
    jblfint16g2 = jblfg2.astype(np.uint16)
    np.save(os.path.join(savepath,newname+num+'g2.npy'),jblfint16g2)
    jblfint16g2.tofile(os.path.join(savepath,newname+num+'g2.raw'))




