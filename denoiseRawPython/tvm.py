import numpy as np
from skimage.restoration import denoise_tv_chambolle
from matplotlib import pylab as plb
import os
import time

path = r'D:\Project\Python\Data\Data\sources\temp\2029\guide'
imagename = 'ceguidecefloat-'
savepath = r'D:\Project\Python\Data\Data\sources\temp\2029\tvm'
savename = 'tvm01-'
weightr = 30
weightb = 40
weightg = 40
epss=2.e-6
n_iter_maxx=10
imgr = np.load(os.path.join(path,imagename+'r.npy'))
imgr  = imgr.astype(np.float64)
imgb = np.load(os.path.join(path,imagename+'b.npy'))
imgb  = imgb.astype(np.float64)
imgg1 = np.load(os.path.join(path,imagename+'g1.npy'))
imgg1  = imgg1.astype(np.float64)
imgg2 = np.load(os.path.join(path,imagename+'g2.npy'))
imgg2  = imgg2.astype(np.float64)

starttime = time.time()
denoise = denoise_tv_chambolle(imgr,weight =weightr,eps = epss,n_iter_max=n_iter_maxx)
endtime = time.time()
print('cost time is',endtime-starttime)
denoise = denoise.astype(np.uint16)
denoise.tofile(os.path.join(savepath,savename+'r.raw'))
np.save(os.path.join(savepath,savename+'r.npy'),denoise)

starttime2 = time.time()
denoise = denoise_tv_chambolle(imgb,weight =weightb,eps = epss,n_iter_max=n_iter_maxx)
endtime2 = time.time()
print('cost time is',endtime2-starttime2)
denoise = denoise.astype(np.uint16)
denoise.tofile(os.path.join(savepath,savename+'b.raw'))
np.save(os.path.join(savepath,savename+'b.npy'),denoise)

starttime3 = time.time()
denoise = denoise_tv_chambolle(imgg1,weight =weightg,eps = epss,n_iter_max=n_iter_maxx)
endtime3 = time.time()
print('cost time is',endtime3-starttime3)
denoise = denoise.astype(np.uint16)
denoise.tofile(os.path.join(savepath,savename+'g1.raw'))
np.save(os.path.join(savepath,savename+'g1.npy'),denoise)

denoise = denoise_tv_chambolle(imgg2,weight =weightg,eps = epss,n_iter_max=n_iter_maxx)
denoise = denoise.astype(np.uint16)
denoise.tofile(os.path.join(savepath,savename+'g2.raw'))
np.save(os.path.join(savepath,savename+'g2.npy'),denoise)


# def denoise_tv_chambolle_nd(image, weight=0.1, eps=2.e-4, n_iter_max=200):
#     """Perform total-variation denoising on n-dimensional images.
#
#     Parameters
#     ----------
#     image : ndarray
#         n-D input data to be denoised.
#     weight : float, optional
#         Denoising weight. The greater `weight`, the more denoising (at
#         the expense of fidelity to `input`).
#     eps : float, optional
#         Relative difference of the value of the cost function that determines
#         the stop criterion. The algorithm stops when:
#
#             (E_(n-1) - E_n) < eps * E_0
#
#     n_iter_max : int, optional
#         Maximal number of iterations used for the optimization.
#
#     Returns
#     -------
#     out : ndarray
#         Denoised array of floats.
#
#     Notes
#     -----
#     Rudin, Osher and Fatemi algorithm.
#     """
#     ndim = image.ndim#(2012,3024)
#     p = np.zeros((image.ndim, ) + image.shape, dtype=image.dtype)#(2,2012,3024)
#     g = np.zeros_like(p)#(2,2012,3024)
#     d = np.zeros_like(image)#(2012,3024)
#     i = 0
#
#     while i < n_iter_max:
#         if i > 0:
#             # d will be the (negative) divergence of p
#             d = -p.sum(0)#(2012,3024)
#             slices_d = [slice(None), ] * ndim#range类似取切片,none代表取全部,start,stop,step
#             slices_p = [slice(None), ] * (ndim + 1)
#             for ax in range(ndim):
#                 slices_d[ax] = slice(1, None)
#                 slices_p[ax+1] = slice(0, -1)
#                 slices_p[0] = ax
#                 d[tuple(slices_d)] += p[tuple(slices_p)]
#                 slices_d[ax] = slice(None)
#                 slices_p[ax+1] = slice(None)
#             out = image + d
#         else:
#             out = image
#         E = (d ** 2).sum()
#
#         # g stores the gradients of out along each axis
#         # e.g. g[0] is the first order finite difference along axis 0
#         slices_g = [slice(None), ] * (ndim + 1)
#         for ax in range(ndim):
#             slices_g[ax+1] = slice(0, -1)
#             slices_g[0] = ax
#             g[tuple(slices_g)] = np.diff(out, axis=ax)
#             slices_g[ax+1] = slice(None)
#
#         norm = np.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...]
#         E += weight * norm.sum()
#         tau = 1. / (2.*ndim)
#         norm *= tau / weight
#         norm += 1.
#         p -= tau * g
#         p /= norm
#         E /= float(image.size)
#         if i == 0:
#             E_init = E
#             E_previous = E
#         else:
#             if np.abs(E_previous - E) < eps * E_init:
#                 break
#             else:
#                 E_previous = E
#         i += 1
#     return out
#
# img = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\guidejbil\guide\DSC00039-01-r.npy')
# img  = img.astype(np.float64)
# result = denoise_tv_chambolle_nd(img)