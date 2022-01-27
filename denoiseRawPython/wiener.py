from scipy.signal import wiener
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab as plb
if __name__ == '__main__':
    path = r"D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\guidejbil\guide\DSC00039-01-g2.npy"  # 90是噪声图
    out = np.load(path)
    out = out.astype(np.float64)

    # 维纳滤波
    #lenaNoise = lenaNoise.astype('float64')
    lenaWiener = wiener(out, [3, 3])
    lenaWiener = np.uint16(lenaWiener / lenaWiener.max() * 16384)

    lenaWiener = lenaWiener.astype(np.uint16)
    #cv2.imwrite(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\wiener3-3.tif', lenaWiener)
    lenaWiener.tofile(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\wiener\DSC00039-01-g2.raw')
    np.save(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\wiener\DSC00039-01-g2.npy', lenaWiener)
    # plt.imshow(lenaWiener, cmap='gray')
    # plt.show()


# #方法一
# from numpy.fft import fft2, ifft2
# from scipy.signal import gaussian
# def gaussian_kernel(kernel_size = 3):
# 	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
# 	h = np.dot(h, h.transpose())
# 	h /= np.sum(h)
# 	return h
# def wiener_filter(img, kernel, K):
# 	kernel /= np.sum(kernel)
# 	dummy = np.copy(img)
# 	dummy = fft2(dummy)
# 	kernel = fft2(kernel, s = img.shape)
# 	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
# 	dummy = dummy * kernel
# 	dummy = np.abs(ifft2(dummy))
# 	return dummy
# path = r"D:\Project\Python\Data\Data\sources\bm3d\DSC01831-gaus500.npy"#90是噪声图
# out = np.load(path)
# out = out.astype(np.float16)
# kernel = gaussian_kernel(5)
# # print(kernel)
# #print(np.abs(kernel))
# filtered_img = wiener_filter(out, kernel, K = 1)
# filtered_img = filtered_img.astype(np.uint16)
# cv2.imwrite(r'D:\Project\Python\Data\Data\sources\wiener\DSC01831-gaus500.tif', filtered_img)
# plb.imshow(filtered_img,'gray')
# plb.show()





#方法二类似opencv
# from PIL import Image
# from scipy import fftpack
# from skimage import img_as_float
# from skimage import img_as_ubyte
# def correlate(original, x):
#     shape = np.array(original.shape)
#
#     fshape = [fftpack.helper.next_fast_len(int(d)) for d in shape]
#     fslice = tuple([slice(0, int(sz)) for sz in shape])
#
#     sp1 = fftpack.fftn(original, fshape)
#     sp2 = fftpack.fftn(x, fshape)
#     ret = fftpack.ifftn(sp1 * sp2)[fslice].copy().real
#     return ret
# # 自适应维纳滤波
# # count 为窗口数，original 为原图
# def adaptiveWienerDeNoise(count, original):
#     count = np.asarray(count)
#     x = np.ones(count)[[slice(None, None, -1)] * np.ones(count).ndim].conj()
#
#     # Estimate the local mean
#     lMean = correlate(original, x) / np.product(count, axis=0)
#
#     # Estimate the local variance
#     lVar = correlate(original ** 2, x) / np.product(count, axis=0) - lMean ** 2
#
#     # Estimate the noise power if needed.
#     noise = np.mean(np.ravel(lVar), axis=0)
#
#     res = (original - lMean)
#     res *= (1 - noise / lVar)
#     res += lMean
#     out = np.where(lVar < noise, lMean, res)
#
#     return out
# imgs = 'D:\Project\Python\Data\Data\sources\others\DSC01831-guas500.jpg'
# original = Image.open(imgs).convert('RGB')
# r,g,b=original.split()
# rDeNoise = Image.fromarray(img_as_ubyte(adaptiveWienerDeNoise([5,5], img_as_float(r))))
# gDeNoise = Image.fromarray(img_as_ubyte(adaptiveWienerDeNoise([5,5], img_as_float(g))))
# bDeNoise = Image.fromarray(img_as_ubyte(adaptiveWienerDeNoise([5,5], img_as_float(b))))
# pic = Image.merge('RGB',[rDeNoise,gDeNoise,bDeNoise])
# plt.imshow(pic)
# plt.show()
# pic.save(imgs[:-4] + '_wiener2_2.png')


# path = r"D:\Project\Python\Data\Data\sources\others\DSC01831-guas500.jpg"#90是噪声图
# out = np.load(path)
# out = out.astype(np.float16)
# denoise = adaptiveWienerDeNoise([5,5], out)
