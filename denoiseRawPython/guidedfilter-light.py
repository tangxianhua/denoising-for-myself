import numpy as np
import cv2
import os
def guideFilter(I, p,Ice,pce, winSize, eps,epsce):
    print(I[0:6, 0:6])
    print(Ice[0:6, 0:6])
    #print(p[0:6, 0:6])
    #print(pce[0:6, 0:6])
    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    mean_p = cv2.blur(p, winSize)      # p的均值平滑
    mean_Ice = cv2.blur(Ice, winSize)      # I的均值平滑
    mean_pce = cv2.blur(pce, winSize)      # p的均值平滑
    # print('********************************')
    #print('mean_I',mean_I[0:6,0:6])
    #print(mean_p[0:6,0:6])
    #print('mean_Ice',mean_Ice[0:6,0:6])
    #print(mean_pce[0:6,0:6])
    # print('********************************')
    mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
    mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑
    mean_IIce = cv2.blur(Ice * Ice, winSize) # I*I的均值平滑
    mean_Ipce = cv2.blur(Ice * pce, winSize) # I*p的均值平滑
    print((I*I)[0:6, 0:6])
    print((Ice*Ice)[0:6, 0:6])
    # print('********************************')
    #print('mean_II',mean_II[0:6,0:6])
    # print(mean_Ip[0:6,0:6])
    #print('mean_IIce',mean_IIce[0:6,0:6])
    # print(mean_Ipce[0:6,0:6])
    # print('********************************')
    var_I = mean_II - mean_I * mean_I  # 方差
    cov_Ip = mean_Ip - mean_I * mean_p # 协方差
    var_Ice = mean_IIce - mean_Ice * mean_Ice  # 方差
    cov_Ipce = mean_Ipce - mean_Ice * mean_pce # 协方差
    #print('mean_I * mean_I',(mean_I * mean_I)[0:6,0:6])
    #print('mean_Ice * mean_Ice',(mean_Ice * mean_Ice)[0:6, 0:6])
    # print('********************************')
    # print(var_I[0:6,0:6])
    # print(var_Ice[0:6,0:6])
    # print(var_I[0:6,0:6]/16383/16383)
    # print(cov_Ip[0:6, 0:6])
    # print(cov_Ipce[0:6, 0:6])
    # print('********************************')
    a = cov_Ip / (var_I + eps)         # 相关因子a
    b = mean_p - a * mean_I            # 相关因子b
    ace = cov_Ipce / (var_Ice + epsce)         # 相关因子a
    bce = mean_pce - ace * mean_Ice            # 相关因子b
    # print('****aaaa****************************')
    # print(a[0:6,0:6])
    # print(b[0:6, 0:6])
    # print('****bbbb****************************')
    mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑
    mean_b = cv2.blur(b, winSize)      # 对b进行均值平滑
    mean_ace = cv2.blur(ace, winSize)      # 对a进行均值平滑
    mean_bce = cv2.blur(bce, winSize)      # 对b进行均值平滑
    # print('*****meana**********************')
    # print(mean_a[0:6,0:6])
    # print(mean_b[0:6, 0:6])
    # print('*****meanb**********************')
    q = mean_a * I + mean_b
    qce = mean_ace * Ice + mean_bce
    # print(q)
    return q

if __name__ == '__main__':
    # epsr = 0.000007
    # epsg = 0.00001
    # epsb = 0.00001
    # winSize = (3, 3)
    # deno = 16383
    # path = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\origin'
    # guideimagepath = r'D:\Project\Python\Data\Data\sources\temp\2029\jbil'
    # savepath = r'D:\Project\Python\Data\Data\sources\temp\guidetest'
    # imagename = 'DSC02029-'
    # guidename = 'c-float-'
    # savename = 'afterNM-'
    #
    # image = np.load(os.path.join(path,imagename+'r.npy'))
    # print('RRRRRRRRRRRRRRRRRRRRRRRRR')
    # I = image / deno
    # bilateral = np.load(os.path.join(guideimagepath,guidename+'r.npy'))
    # bilateral = bilateral / deno
    # guideFilter_img = guideFilter(I, bilateral, winSize, epsr)
    # guideFilter_img = guideFilter_img * deno
    # guideFilter_img[guideFilter_img > deno] = deno
    # guideFilter_img = np.round(guideFilter_img)
    # guideFilter_img = guideFilter_img.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'r.npy'),guideFilter_img)
    # guideFilter_img.tofile(os.path.join(savepath,savename+'r.raw'))
    #
    # image = np.load(os.path.join(path,imagename+'b.npy'))
    # print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
    # I = image / deno
    # bilateral = np.load(os.path.join(guideimagepath,guidename+'b.npy'))
    # bilateral = bilateral / deno
    # guideFilter_img = guideFilter(I, bilateral, winSize, epsb)
    # guideFilter_img = guideFilter_img * deno
    # guideFilter_img[guideFilter_img > deno] = deno
    # guideFilter_img = np.round(guideFilter_img)
    # guideFilter_img = guideFilter_img.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'b.npy'),guideFilter_img)
    # guideFilter_img.tofile(os.path.join(savepath,savename+'b.raw'))
    #
    # image = np.load(os.path.join(path,imagename+'g1.npy'))
    # print('G1G11G11111111111111111111111111111111')
    # I = image / deno
    # bilateral = np.load(os.path.join(guideimagepath,guidename+'g1.npy'))
    # bilateral = bilateral / deno
    # guideFilter_img = guideFilter(I, bilateral, winSize, epsg)
    # guideFilter_img = guideFilter_img * deno
    # guideFilter_img[guideFilter_img > deno] = deno
    # guideFilter_img = np.round(guideFilter_img)
    # guideFilter_img = guideFilter_img.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'g1.npy'),guideFilter_img)
    # guideFilter_img.tofile(os.path.join(savepath,savename+'g1.raw'))
    #
    # image = np.load(os.path.join(path,imagename+'g2.npy'))
    # print('G222222222222222222222222222222222222')
    # I = image / deno
    # bilateral = np.load(os.path.join(guideimagepath,guidename+'g2.npy'))
    # bilateral = bilateral / deno
    # guideFilter_img = guideFilter(I, bilateral, winSize, epsg)
    # guideFilter_img = guideFilter_img * deno
    # guideFilter_img[guideFilter_img > deno] = deno
    # guideFilter_img = np.round(guideFilter_img)
    # guideFilter_img = guideFilter_img.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'g2.npy'),guideFilter_img)
    # guideFilter_img.tofile(os.path.join(savepath,savename+'g2.raw'))

    epsr = 1878.81#0.000007 *16383*16383
    epsg = 2684#0.00001
    epsb = 2684#0.00001
    epsrce = 0.000007
    epsgce = 0.00001
    epsbce = 0.00001
    deno = 16383
    winSize = (3, 3)
    path = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\origin'
    guideimagepath = r'D:\Project\Python\Data\Data\sources\temp\2029\jbil'
    savepath = r'D:\Project\Python\Data\Data\sources\temp\guidetest'
    imagename = 'DSC02029-'
    guidename = 'c-float-'
    savename = 'withoutNM-'

    image_R = np.load(os.path.join(path,imagename+'r.npy'))
    print('RRRRRRRRRRRRRRRRRRRRRRRRR')
    image_Rce = image_R/deno
    bilateral_R = np.load(os.path.join(guideimagepath,guidename+'r.npy'))
    bilateral_Rce = bilateral_R/deno
    guideFilter_R = guideFilter(image_R, bilateral_R, image_Rce, bilateral_Rce,winSize, epsr,epsrce)
    guideFilter_R[guideFilter_R > 16383] = 16383
    guideFilter_R = np.round(guideFilter_R)
    image_B = np.load(os.path.join(path,imagename+'b.npy'))
    # print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
    # bilateral_B = np.load(os.path.join(guideimagepath,guidename+'b.npy'))
    # guideFilter_B = guideFilter(image_B, bilateral_B, winSize, epsb)
    # guideFilter_B[guideFilter_B > 16383] = 16383
    # guideFilter_B = np.round(guideFilter_B)
    # image_G1 = np.load(os.path.join(path,imagename+'g1.npy'))
    # print('G1G11G11111111111111111111111111111111')
    # bilateral_G1 = np.load(os.path.join(guideimagepath,guidename+'g1.npy'))
    # guideFilter_G1 = guideFilter(image_G1, bilateral_G1, winSize, epsg)
    # guideFilter_G1[guideFilter_G1 > 16383] = 16383
    # guideFilter_G1 = np.round(guideFilter_G1)
    # image_G2 = np.load(os.path.join(path,imagename+'g2.npy'))
    # print('G222222222222222222222222222222222222')
    # bilateral_G2 = np.load(os.path.join(guideimagepath,guidename+'g2.npy'))
    # guideFilter_G2 = guideFilter(image_G2, bilateral_G2, winSize, epsg)
    # guideFilter_G2[guideFilter_G2 > 16383] = 16383
    # guideFilter_G2 = np.round(guideFilter_G2)



    # guideFilter_R = guideFilter_R.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'r.npy'),guideFilter_R)
    # guideFilter_R.tofile(os.path.join(savepath,savename+'r.raw'))
    # guideFilter_B = guideFilter_B.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'b.npy'),guideFilter_B)
    # guideFilter_B.tofile(os.path.join(savepath,savename+'b.raw'))
    # guideFilter_G1 = guideFilter_G1.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'g1.npy'),guideFilter_G1)
    # guideFilter_G1.tofile(os.path.join(savepath,savename+'g1.raw'))
    # guideFilter_G2 = guideFilter_G2.astype(np.uint16)
    # np.save(os.path.join(savepath,savename+'g2.npy'),guideFilter_G2)
    # guideFilter_G2.tofile(os.path.join(savepath,savename+'g2.raw'))