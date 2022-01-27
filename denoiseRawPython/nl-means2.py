import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import cv2
import sys
from tqdm import tqdm
#np.random.seed(1)
import os



# def gauss(kernel_size, sigma):
#     kernel = np.zeros((kernel_size, kernel_size))
#     center = kernel_size // 2
#     if sigma <= 0:
#         sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
#
#     s = sigma ** 2
#     sum_val = 0
#     for i in range(kernel_size):
#         for j in range(kernel_size):
#             x, y = i - center, j - center
#
#             kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
#             sum_val += kernel[i, j]
#
#     kernel = kernel / sum_val
#
#     return kernel
# gauss_kernel = gauss(5, 2)


def NLmeans(img, kernel_cov,filter_size,search_size,deno):
    width, height = img.shape
    pad_img = np.pad(img, ((filter_size, filter_size), (filter_size, filter_size)), 'symmetric')#在图像边缘填充pad

    result = np.zeros(img.shape)#原图全0 copy
    kernel = np.ones((2 * filter_size + 1, 2 * filter_size + 1))
    kernel = kernel / ((2 * filter_size + 1) ** 2)
    #kernel = gauss_kernel

    pbar = tqdm(total=width * height)
    for w in range(width):
        for h in range(height):
            pbar.update(1)
            w1 = w + filter_size#
            h1 = h + filter_size#
            x_pixels = pad_img[w1-filter_size:w1+filter_size+1, h1-filter_size:h1+filter_size+1]#x和y的尺寸都是2*filtersize+1
            w_min = max(w1-search_size, filter_size)#防止越界
            w_max = min(w1+search_size, width+filter_size-1)
            h_min = max(h1-search_size, filter_size)
            h_max = min(h1+search_size, height+filter_size-1)
            sum_similarity = 0
            sum_pixel = 0
            weight_max = 0
            for x in range(w_min, w_max+1):#serch size
                for y in range(h_min, h_max+1):
                    if (x == w1) and (y == h1):
                        continue
                    y_pixels = pad_img[x-filter_size:x+filter_size+1, y-filter_size:y+filter_size+1]
                    distance = x_pixels - y_pixels

                    distance = np.sum(np.multiply(kernel, np.square(distance)))/deno#可以除
                    similarity = np.exp(-distance/(kernel_cov*kernel_cov))#kernel_cov为平滑参数，越小保留细节越多

                    if similarity > weight_max:
                        weight_max = similarity
                    sum_similarity += similarity
                    sum_pixel += similarity * pad_img[x, y]
            sum_pixel += weight_max * pad_img[w1, h1]
            sum_similarity += weight_max
            if sum_similarity > 0:
                result[w, h] = sum_pixel / sum_similarity
            else:
                result[w, h] = img[w, h]
    pbar.close()
    return result

if __name__ == "__main__":
    # path = r"D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000.npy"
    # out = np.load(path)
    # out2 = np.squeeze(out)
    # out2 = out2.astype(np.float32)
    # #starttime = time.time()
    # denoised_img = NLmeans(out2, 5)
    # #endtime = time.time()
    # #print("time cost", starttime-endtime)
    # save = denoised_img.astype(np.uint16)
    # np.save(r'D:\Project\Python\Data\Data\sources\bm3d\gaus300\DSC01831-gaus300-sig1000-nlmean5.npy', denoised_img)
    # denoised_img.tofile('.raw')

    path = r"D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\origin"
    savepath = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\nlmeans'
    imagename = 'DSC02028-'
    h = 5
    filter_size = 3  # the radio of the filter
    search_size = 15  # the ratio of the search size
    deno = 32
    print(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-r.npy'))
    print(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-r.raw'))
    out = np.load(os.path.join(path,imagename+'r.npy'))
    out2 = np.squeeze(out)
    out2 = out2.astype(np.float32)
    denoised_img = NLmeans(out2, h,filter_size,search_size,deno)
    save = denoised_img.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-r.npy'), save)
    save.tofile(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-r.raw'))

    print(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-b.npy'))
    print(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-b.raw'))
    out = np.load(os.path.join(path,imagename+'b.npy'))
    out2 = np.squeeze(out)
    out2 = out2.astype(np.float32)
    denoised_img = NLmeans(out2, h,filter_size,search_size,deno)
    save = denoised_img.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-b.npy'), save)
    save.tofile(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-b.raw'))

    print(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-g1.npy'))
    print(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-g1.raw'))
    out = np.load(os.path.join(path,imagename+'g1.npy'))
    out2 = np.squeeze(out)
    out2 = out2.astype(np.float32)
    denoised_img = NLmeans(out2, h,filter_size,search_size,deno)
    save = denoised_img.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-g1.npy'), save)
    save.tofile(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-g1.raw'))

    print(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-g2.npy'))
    print(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-g2.raw'))
    out = np.load(os.path.join(path,imagename+'g2.npy'))
    out2 = np.squeeze(out)
    out2 = out2.astype(np.float32)
    denoised_img = NLmeans(out2, h,filter_size,search_size,deno)
    save = denoised_img.astype(np.uint16)
    np.save(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-g2.npy'), save)
    save.tofile(os.path.join(savepath,imagename+'nlmeans'+'-01'+'-g2.raw'))