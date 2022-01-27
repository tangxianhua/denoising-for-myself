import numpy as np
from PIL import Image
from scipy.signal import convolve2d


# def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
#     """
#     2D gaussian mask - should give the same result as MATLAB's
#     fspecial('gaussian',[shape],[sigma])
#     """
#     m, n = [(ss - 1.) / 2. for ss in shape]
#     y, x = np.ogrid[-m:m + 1, -n:n + 1]
#     h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0
#     sumh = h.sum()
#     if sumh != 0:
#         h /= sumh
#     return h
#
#
# def filter2(x, kernel, mode='same'):
#     return convolve2d(x, np.rot90(kernel, 2), mode=mode)
#
#
# def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=18000):
#     if not im1.shape == im2.shape:
#         raise ValueError("Input Imagees must have the same dimensions")
#     if len(im1.shape) > 2:
#         raise ValueError("Please input the images with 1 channel")
#
#     M, N = im1.shape
#     C1 = (k1 * L) ** 2
#     C2 = (k2 * L) ** 2
#     window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
#     window = window / np.sum(np.sum(window))
#
#     if im1.dtype == np.uint8:
#         im1 = np.double(im1)
#     if im2.dtype == np.uint8:
#         im2 = np.double(im2)
#
#     mu1 = filter2(im1, window, 'valid')
#     mu2 = filter2(im2, window, 'valid')
#     mu1_sq = mu1 * mu1
#     mu2_sq = mu2 * mu2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
#     sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
#     sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
#
#     return np.mean(np.mean(ssim_map))
#
#
# if __name__ == "__main__":
#     # im1 = Image.open("1.png")
#     # im2 = Image.open("2.png")
#     im1 = np.load(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000.npy')
#     im2 = np.load(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000-jbil.npy')
#
#
#     print(compute_ssim(np.array(im1), np.array(im2)))

#SSIM计算
# def ssim(y_true , y_pred):
#     u_true = np.mean(y_true)
#     u_pred = np.mean(y_pred)
#     var_true = np.var(y_true)
#     var_pred = np.var(y_pred)
#     std_true = np.sqrt(var_true)
#     std_pred = np.sqrt(var_pred)
#     c1 = np.square(0.01*7)
#     c2 = np.square(0.03*7)
#     ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
#     denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
#     return ssim/denom
# img1 = np.load(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-300.npy')
# img2 = np.load(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000-nlmean5.npy')
#
# result = ssim(img1,img2)
# print(result)

#mse计算
# def mse(img1, img2):
#     mse = np.mean( (img1 - img2) ** 2 )
#     return mse
# img1 = np.load(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-300.npy')
# img2 = np.load(r'D:\Project\Python\Data\Data\sources\bm3d\bm3dmy2\DSC01831-gaus300-sig1000-s2-pro.npy')
# result = mse(img1,img2)
# print(result)


#PSNR计算
# import sys
# def ComputePSNR(Img1, Img2):
#     """
#     Compute the Peak Signal to Noise Ratio (PSNR) in decibles(dB).
#     """
#     if Img1.size != Img2.size:
#         print('ERROR: two images should be in same size in computing PSNR.\n')
#         sys.exit()
#
#     Img1 = Img1.astype(np.float64)
#     Img2 = Img2.astype(np.float64)
#     RMSE = np.sqrt(np.sum((Img1 - Img2) ** 2) / Img1.size)
#     return 20 * np.log10(18000. / RMSE)
#
path = r"D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\origin\DSC02030.npy"
# out = np.load(path)
# out2 = np.squeeze(out)
# out2 = out2.astype(np.uint16)
# path2 = r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000-guide.npy'
# nlmean = np.load(path2)
# final_PSNR = ComputePSNR(out2, nlmean)
# print('The PSNR of final image is {} dB.\n'.format(final_PSNR))
