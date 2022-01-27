import numpy as np
import pandas as pd
import os
import math
#mse计算
def mse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse
def sn(S,MSE):
    SN = 20*(np.log10(S/MSE))
    return SN
def PSNR(maxvalue,mse):
    return 20 * np.log10(maxvalue / mse)

list38rgb = [[2312,2416,3182,3285],[2458,2556,3185,3283],[2597,2704,3183,3289],[2739,2844,3182,3290],
             [2308,2422,3319,3434],[2455,2558,3325,3432],[2591,2702,3322,3432],[2735,2839,3325,3431],
             [2312,2412,3468,3568],[2453,2555,3468,3569],[2593,2698,3469,3576],[2733,2840,3468,3571],
             [2309,2412,3608,3711],[2448,2551,3606,3711],[2591,2697,3608,3711],[2735,2837,3610,3717],
             [2305,2411,3743,3846],[2444,2549,3746,3848],[2588,2693,3750,3850],[2732,2832,3752,3850],
             [2307,2400,3882,3990],[2448,2546,3882,3985],[2588,2683,3883,3983],[2728,2821,3889,3989]]
list39rgb = [[2312,2416,3179,3285],[2453,2561,3182,3290],[2596,2706,3182,3291],[2738,2846,3183,3288],[2313,2416,3323,3427],[2452,2561,3324,3431],
                [2596,2703,3327,3433],[2738,2846,3326,3434],[2312,2413,3467,3566],[2451,2557,3466,3571],[2594,2700,3469,3573],[2737,2840,3472,3574],
                [2311,2412,3609,3708],[2451,2554,3610,3712],[2593,2696,3610,3714],[2734,2836,3612,3713],[2312,2410,3748,3842],[2453,2550,3749,3846],
                [2590,2692,3753,3849],[2734,2831,3747,3852],[2309,2406,3880,3966],[2455,2556,3884,3984],[2599,2691,3886,3979],[2733,2820,3892,3984]]
list41rgb = [[1585,1686,2093,2185],[1728,1829,2093,2188],[1877,1971,2096,2188],[2015,2109,2098,2192],
             [1583,1689,2226,2330],[1734,1831,2231,2330],[1875,1976,2235,2334],[2017,2116,2238,2334],
             [1587,1691,2367,2475],[1732,1831,2373,2474],[1874,1976,2373,2475],[2015,2113,2376,2477],
             [1583,1689,2511,2621],[1732,1829,2516,2619],[1874,1973,2516,2621],[2017,2114,2520,2619],
             [1585,1689,2660,2766],[1732,1829,2665,2760],[1872,1974,2663,2762],[2017,2111,2667,2762],
             [1585,1686,2805,2909],[1728,1826,2805,2909],[1872,1971,2805,2907],[2015,2109,2801,2906]]

list42rgb = [[1589,1692,2092,2187],[1735,1832,2097,2189],[1877,1974,2097,2192],[2019,2114,2101,2198],
             [1591,1695,2227,2329],[1735,1837,2234,2331],[1881,1979,2236,2335],[2021,2114,2239,2336],
             [1593,1695,2371,2473],[1737,1836,2372,2471],[1881,1976,2376,2475],[2019,2113,2380,2480],
             [1593,1693,2516,2619],[1737,1836,2518,2617],[1879,1974,2520,2621],[2021,2114,2520,2621],
             [1593,1690,2658,2763],[1737,1832,2657,2761],[1877,1972,2664,2757],[2023,2111,2664,2764],
             [1593,1681,2804,2910],[1735,1823,2806,2899],[1875,1969,2802,2907],[2014,2109,2804,2905]]

list38sony = [[3164,2288,3258,2387],[3166,2429,3261,2523],[3163,2560,3270,2669],[3163,2697,3272,2808],
             [3294,2283,3404,2392],[3294,2420,3405,2533],[3296,2559,3409,2670],[3298,2698,3410,2808],
             [3431,2282,3538,2392],[3434,2422,3543,2529],[3436,2558,3546,2667],[3436,2697,3547,2805],
             [3567,2284,3674,2389],[3570,2420,3678,2527],[3572,2557,3683,2665],[3573,2695,3684,2803],
             [3702,2282,3807,2388],[3705,2419,3814,2526],[3709,2555,3817,2662],[3712,2693,3817,2799],
             [3835,2282,3952,2383],[3844,2417,3945,2519],[3848,2555,3949,2652],[3849,2693,3954,2796]]
list38my= [[3176,2309,3289,2418],[3180,2454,3290,2562],[3180,2595,3292,2706],[3183,2739,3292,2845],
             [3319,2309,3428,2417],[3324,2454,3433,2559],[3324,2596,3433,2701],[3327,2739,3435,2844],
             [3464,2308,3571,2414],[3465,2452,3572,2558],[3469,2596,3574,2697],[3471,2737,3575,2840],
             [3605,2310,3707,2411],[3610,2451,3713,2553],[3609,2592,3715,2694],[3610,2735,3716,2837],
             [3743,2306,3844,2409],[3745,2448,3849,2551],[3749,2588,3850,2692],[3752,2731,3851,2829],
             [3877,2303,3995,2405],[3884,2440,3989,2545],[3884,2584,3989,2683],[3891,2725,3990,2827]]


listO = list38sony
listN = list38my
#listD = list42rgb

# path = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\42vs41\sigmatest'
# imagenumO = 'RGB_out41-'#原图
# imagenumN = 'RGB_out42-'
# imagenumD = 'RGB_out42-01-'
csvsavepath = r'D:\Project\Python\Data\Data\sources\temp'
maxvalue = 65535

# origin = np.load(os.path.join(path,imagenumO+'r.npy'))
# noise = np.load(os.path.join(path,imagenumN+'r.npy'))
#denoise = np.load(os.path.join(path,imagenumD+'r.npy'))
import cv2
origin = cv2.imread(r'D:\Project\Python\Data\Data\sources\temp\DSC00039.JPG')[:,:,1]
noise =cv2.imread(r'D:\Project\Python\Data\Data\sources\temp\RGB_out2021-12-20-14-07-41-138(1).jpg')[:,:,1]
SNlistO = []
SNlistN = []
# SNlistD = []
# PSNRlistO = []
# PSNRlistN = []
# PSNRlistD = []

for i in range(24):
    chartRegionO = origin[listO[i][1]:listO[i][3],listO[i][0]:listO[i][2]]
    #print(chartRegionO)
    S = np.linalg.norm(chartRegionO)
    msesn = np.linalg.norm(chartRegionO - np.mean(chartRegionO))
    snresult = sn(S,msesn)
    SNlistO.append(snresult)
    # psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionO))
    # PSNRlistO.append(psnrresult)
#
    chartRegionN = noise[listN[i][1]:listN[i][3],listN[i][0]:listN[i][2]]
    S = np.linalg.norm(chartRegionN)
    msesn = np.linalg.norm(chartRegionN - np.mean(chartRegionN))
    snresult = sn(S,msesn)
    SNlistN.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionN))
#     PSNRlistN.append(psnrresult)
#
#     chartRegionD = denoise[listD[i][0]:listD[i][1],listD[i][2]:listD[i][3]]
#     S = np.linalg.norm(chartRegionD)
#     msesn = np.linalg.norm(chartRegionD - np.mean(chartRegionD))
#     snresult = sn(S,msesn)
#     SNlistD.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionD))
#     PSNRlistD.append(psnrresult)
# listall = [SNlistO,SNlistN,SNlistD,PSNRlistO,PSNRlistN,PSNRlistD]
# listcsv = pd.DataFrame(data=listall,index = ['SNorigin', 'SNnoise', 'SNdenoise','PSNRorigin','PSNRnoise','PSNRdenoise'])
# print('保存路径',os.path.join(csvsavepath,imagenumD+'r.csv'))
# listcsv.to_csv(os.path.join(csvsavepath,imagenumD+'r.csv'))

listall = [SNlistO,SNlistN]
listcsv = pd.DataFrame(data=listall,index = ['SNorigin', 'SNnoise'])
imagename = 'temp'
print('保存路径',os.path.join(csvsavepath,imagename+'.csv'))
listcsv.to_csv(os.path.join(csvsavepath,imagename+'.csv'))


#
# origin = np.load(os.path.join(path,imagenumO+'b.npy'))
# noise = np.load(os.path.join(path,imagenumN+'b.npy'))
# denoise = np.load(os.path.join(path,imagenumD+'b.npy'))
# SNlistO = []
# SNlistN = []
# SNlistD = []
# PSNRlistO = []
# PSNRlistN = []
# PSNRlistD = []
# for i in range(24):
#     chartRegionO = origin[listO[i][0]:listO[i][1],listO[i][2]:listO[i][3]]
#     S = np.linalg.norm(chartRegionO)
#     msesn = np.linalg.norm(chartRegionO - np.mean(chartRegionO))
#     snresult = sn(S,msesn)
#     SNlistO.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionO))
#     PSNRlistO.append(psnrresult)
#
#     chartRegionN = noise[listN[i][0]:listN[i][1],listN[i][2]:listN[i][3]]
#     S = np.linalg.norm(chartRegionN)
#     msesn = np.linalg.norm(chartRegionN - np.mean(chartRegionN))
#     snresult = sn(S,msesn)
#     SNlistN.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionN))
#     PSNRlistN.append(psnrresult)
#
#     chartRegionD = denoise[listD[i][0]:listD[i][1],listD[i][2]:listD[i][3]]
#     S = np.linalg.norm(chartRegionD)
#     msesn = np.linalg.norm(chartRegionD - np.mean(chartRegionD))
#     snresult = sn(S,msesn)
#     SNlistD.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionD))
#     PSNRlistD.append(psnrresult)
# listall = [SNlistO,SNlistN,SNlistD,PSNRlistO,PSNRlistN,PSNRlistD]
# listcsv = pd.DataFrame(data=listall,index = ['SNorigin', 'SNnoise', 'SNdenoise','PSNRorigin','PSNRnoise','PSNRdenoise'])
# print('保存路径',os.path.join(csvsavepath,imagenumD+'b.csv'))
# listcsv.to_csv(os.path.join(csvsavepath,imagenumD+'b.csv'))
#
# origin = np.load(os.path.join(path,imagenumO+'g.npy'))
# noise = np.load(os.path.join(path,imagenumN+'g.npy'))
# denoise = np.load(os.path.join(path,imagenumD+'g.npy'))
# SNlistO = []
# SNlistN = []
# SNlistD = []
# PSNRlistO = []
# PSNRlistN = []
# PSNRlistD = []
# for i in range(24):
#     chartRegionO = origin[listO[i][0]:listO[i][1],listO[i][2]:listO[i][3]]
#     S = np.linalg.norm(chartRegionO)
#     msesn = np.linalg.norm(chartRegionO - np.mean(chartRegionO))
#     snresult = sn(S,msesn)
#     SNlistO.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionO))
#     PSNRlistO.append(psnrresult)
#
#     chartRegionN = noise[listN[i][0]:listN[i][1],listN[i][2]:listN[i][3]]
#     S = np.linalg.norm(chartRegionN)
#     msesn = np.linalg.norm(chartRegionN - np.mean(chartRegionN))
#     snresult = sn(S,msesn)
#     SNlistN.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionN))
#     PSNRlistN.append(psnrresult)
#
#     chartRegionD = denoise[listD[i][0]:listD[i][1],listD[i][2]:listD[i][3]]
#     S = np.linalg.norm(chartRegionD)
#     msesn = np.linalg.norm(chartRegionD - np.mean(chartRegionD))
#     snresult = sn(S,msesn)
#     SNlistD.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionD))
#     PSNRlistD.append(psnrresult)
# listall = [SNlistO,SNlistN,SNlistD,PSNRlistO,PSNRlistN,PSNRlistD]
# listcsv = pd.DataFrame(data=listall,index = ['SNorigin', 'SNnoise', 'SNdenoise','PSNRorigin','PSNRnoise','PSNRdenoise'])
# print('保存路径',os.path.join(csvsavepath,imagenumD+'g.csv'))
# listcsv.to_csv(os.path.join(csvsavepath,imagenumD+'g.csv'))
#


