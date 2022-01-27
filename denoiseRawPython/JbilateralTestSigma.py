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

list16=[[1043,1099,1366,1428],[1042,1097,1442,1502],[1040,1096,1516,1576],[1039,1094,1592,1648],[1038,1094,1665,1721],[1038,1092,1737,1791],
                [1113,1169,1369,1430],[1111,1167,1445,1505],[1109,1166,1522,1580],[1108,1164,1597,1654],[1106,1162,1671,1726],[1107,1160,1745,1799],
                [1184,1241,1371,1433],[1181,1240,1448,1509],[1179,1236,1525,1585],[1177,1234,1602,1660],[1177,1232,1678,1733],[1174,1230,1753,1806],
                [1258,1317,1372,1436],[1254,1311,1450,1512],[1251,1308,1529,1589],[1248,1307,1607,1666],[1246,1304,1684,1741],[1244,1295,1758,1812]]
list17 = [[1048,1096,1370,1426],[1044,1097,1444,1500],[1043,1096,1518,1574],[1042,1094,1592,1647],[1041,1092,1665,1720],[1039,1093,1737,1791],
              [1113,1169,1373,1428],[1111,1167,1446,1504],[1109,1165,1521,1580],[1108,1163,1597,1652],[1107,1161,1671,1726],[1107,1160,1745,1797],
              [1185,1243,1372,1432],[1183,1240,1448,1508],[1181,1237,1526,1585],[1178,1234,1603,1659],[1177,1231,1678,1733],[1175,1231,1754,1807],
              [1258,1317,1374,1434],[1256,1313,1453,1510],[1253,1309,1530,1588],[1249,1307,1607,1665],[1247,1304,1683,1740],[1245,1293,1759,1815]]

list24=[[994,1025,1735,1767],[993,1023,1779,1811],[992,1023,1822,1853],[991,1022,1866,1898],[990,1022,1909,1942],[990,1021,1954,1985],
                [1036,1066,1734,1766],[1036,1065,1779,1810],[1035,1065,1822,1853],[1034,1064,1866,1897],[1034,1064,1910,1941],[1034,1064,1954,1987],
                [1079,1109,1734,1765],[1078,1109,1778,1808],[1077,1108,1822,1853],[1077,1108,1865,1898],[1076,1108,1910,1942],[1076,1108,1954,1986],
                [1121,1152,1733,1764],[1120,1152,1777,1808],[1120,1152,1821,1853],[1120,1152,1866,1898],[1120,1152,1910,1942],[1120,1152,1954,1987]]
list25=[[1001,1026,1740,1767],[994,1024,1778,1810],[992,1023,1821,1853],[992,1023,1866,1897],[991,1022,1909,1941],[991,1022,1953,1985],
                [1037,1067,1734,1765],[1037,1067,1778,1809],[1036,1067,1822,1853],[1035,1065,1865,1897],[1033,1065,1908,1942],[1032,1064,1953,1986],
                [1079,1109,1733,1765],[1078,1110,1777,1808],[1077,1109,1821,1853],[1077,1108,1865,1898],[1077,1109,1909,1941],[1078,1109,1953,1987],
                [1121,1153,1732,1764],[1121,1153,1776,1808],[1120,1153,1820,1853],[1120,1153,1866,1898],[1120,1153,1908,1941],[1120,1152,1955,1986]]
list32=[[1192,1300,1926,2035],[1189,1299,2064,2170],[1187,1296,2196,2304],[1185,1291,2331,2435],[1182,1287,2465,2565],[1178,1283,2593,2690],
                [1328,1433,1929,2035],[1325,1433,2065,2171],[1323,1429,2199,2305],[1321,1425,2334,2437],[1318,1422,2467,2568],[1312,1418,2595,2697],
                [1462,1567,1931,2036],[1460,1567,2065,2171],[1458,1565,2198,2306],[1457,1560,2335,2438],[1453,1555,2470,2569],[1450,1550,2602,2700],
                [1595,1698,1929,2036],[1594,1697,2065,2170],[1592,1697,2199,2305],[1589,1695,2333,2439],[1587,1687,2468,2567],[1583,1681,2603,2698]]
list31=[[1191,1300,1929,2036],[1189,1299,2064,2172],[1186,1296,2199,2305],[1184,1292,2332,2437],[1180,1288,2465,2567],[1179,1284,2597,2691],
                [1328,1435,1932,2037],[1325,1434,2066,2174],[1322,1431,2203,2306],[1318,1427,2335,2441],[1316,1422,2468,2572],[1312,1418,2599,2699],
                [1461,1566,1931,2037],[1459,1567,2068,2173],[1457,1564,2201,2307],[1454,1561,2336,2441],[1450,1557,2467,2574],[1445,1551,2599,2702],
                [1592,1698,1931,2034],[1591,1698,2064,2172],[1590,1697,2198,2306],[1589,1693,2334,2440],[1585,1689,2469,2573],[1580,1683,2603,2699]]
list38rgb = [[2312,2416,3179,3285],[2453,2561,3182,3290],[2596,2706,3182,3291],[2738,2846,3183,3288],[2313,2416,3323,3427],[2452,2561,3324,3431],
                [2596,2703,3327,3433],[2738,2846,3326,3434],[2312,2413,3467,3566],[2451,2557,3466,3571],[2594,2700,3469,3573],[2737,2840,3472,3574],
                [2311,2412,3609,3708],[2451,2554,3610,3712],[2593,2696,3610,3714],[2734,2836,3612,3713],[2312,2410,3748,3842],[2453,2550,3749,3846],
                [2590,2692,3753,3849],[2734,2831,3747,3852],[2309,2406,3880,3966],[2455,2556,3884,3984],[2599,2691,3886,3979],[2733,2820,3892,3984]]

list16 = list38rgb
list17 = list38rgb

# pathorigin = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\origin'
# pathnoise = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\16vs17\origin'
# pathdenoise = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\16vs17\guide-jbil'
# imagenumiso100 = 'DSC00016-'#原图
# imagenumiso6400 = 'DSC00017-'#噪声图
# imagenumdenoise = 'DSC00017-13-'#去噪图
# csvsavepath = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\24vs25vs26\25\csv\newcsv'
# R16 = np.load(os.path.join(pathorigin,imagenumiso100+'r.npy'))
# R17 = np.load(os.path.join(pathnoise,imagenumiso6400+'r.npy'))
# Denoise = np.load(os.path.join(pathdenoise,imagenumdenoise+'r.npy'))
# maxvalue = 16383
R16 = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\origin\DSC00039-r.npy')
R17 = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\origin\DSC00039-r.npy')
Denoise = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\38vs39\origin\DSC00039-r.npy')

stdlist16 = []#标准差
stdlist17 = []
stdlistdenoise = []
meanlist16 = []#均值
meanlist17 = []
# medianlist16 = []#中值
# medianlist17 = []
# maxlist16 = []#最大值
# maxlist17 = []
# minlist16 = []#最小值
# minlist17 = []
# absdiff = []
msenoiselist = []
msedenoiselist = []
SNlist = []
PSNRlist = []
for i in range(24):

    chartRegion16 = R16[list16[i][0]:list16[i][1],list16[i][2]:list16[i][3]]
    chartRegion17 = R17[list17[i][0]:list17[i][1], list17[i][2]:list17[i][3]]
    chartRegionDeoise = Denoise[list17[i][0]:list17[i][1], list17[i][2]:list17[i][3]]
    #统一模块大小
    list1= [chartRegion16.shape[0],chartRegion17.shape[0]]
    list2 = [chartRegion16.shape[1],chartRegion17.shape[1]]
    list1 = np.array(list1)
    list2 = np.array(list2)
    height = list1.min()
    width = list2.min()
    chartRegion16 = chartRegion16[0:height,0:width]
    chartRegion17 = chartRegion17[0:height,0:width]
    chartRegionDeoise = chartRegionDeoise[0:height, 0:width]

    stdlist16.append(round(chartRegion16.std()))#标准差
    meanlist16.append(round(np.mean(chartRegion16)))#均值
    # medianlist16.append(np.median(chartRegion16))#中值
    # maxlist16.append(chartRegion16.max())  # 最大值
    # minlist16.append(chartRegion16.min())  # 最小值
    # absdiff.append(round(abs(chartRegion17.std()-chartRegion16.std())))

    stdlist17.append(round(chartRegion17.std()))#标准差
    mseNoiseResult = mse(chartRegion16,chartRegion17)
    msenoiselist.append(mseNoiseResult)
    stdlistdenoise.append(round(chartRegionDeoise.std()))  # 标准差
    msedenoiseResult = mse(chartRegion16,chartRegionDeoise)
    msedenoiselist.append(msedenoiseResult)

    S = np.linalg.norm(chartRegion16)
    msesn = np.linalg.norm(chartRegionDeoise - np.mean(chartRegion16))
    snresult = sn(S,msesn)
    print(snresult)
    # print(np.mean(chartRegion16))
    # print(chartRegionDeoise)
    # print(chartRegionDeoise - np.mean(chartRegion16))
    # print(S)
    # print(msesn)
    # SNlist.append(snresult)
    # psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionDeoise))
    # PSNRlist.append(psnrresult)
#
# listall = [stdlist16,stdlist17,stdlistdenoise,meanlist16,SNlist,PSNRlist,msenoiselist,msedenoiselist]
# listcsv = pd.DataFrame(data=listall)
# print('保存路径',os.path.join(csvsavepath,imagenumdenoise+'r.csv'))
# listcsv.to_csv(os.path.join(csvsavepath,imagenumdenoise+'r.csv'))


#
# stdlist16 = []#标准差
# stdlist17 = []
# stdlistdenoise = []
# meanlist16 = []#均值
# meanlist17 = []
# msenoiselist = []
# msedenoiselist = []
# SNlist = []
# PSNRlist = []
#
# R16 = np.load(os.path.join(pathorigin,imagenumiso100+'b.npy'))
# R17 = np.load(os.path.join(pathnoise,imagenumiso6400+'b.npy'))
# Denoise = np.load(os.path.join(pathdenoise,imagenumdenoise+'b.npy'))
#
# for i in range(24):
#     chartRegion16 = R16[list16[i][0]:list16[i][1],list16[i][2]:list16[i][3]]
#     chartRegion17 = R17[list17[i][0]:list17[i][1], list17[i][2]:list17[i][3]]
#     chartRegionDeoise = Denoise[list17[i][0]:list17[i][1], list17[i][2]:list17[i][3]]
#     #统一模块大小
#     list1= [chartRegion16.shape[0],chartRegion17.shape[0]]
#     list2 = [chartRegion16.shape[1],chartRegion17.shape[1]]
#     list1 = np.array(list1)
#     list2 = np.array(list2)
#     height = list1.min()
#     width = list2.min()
#     chartRegion16 = chartRegion16[0:height,0:width]
#     chartRegion17 = chartRegion17[0:height,0:width]
#     chartRegionDeoise = chartRegionDeoise[0:height, 0:width]
#
#     stdlist16.append(round(chartRegion16.std()))#标准差
#     meanlist16.append(round(np.mean(chartRegion16)))#均值
#
#     stdlist17.append(round(chartRegion17.std()))#标准差
#     mseNoiseResult = mse(chartRegion16,chartRegion17)
#     msenoiselist.append(mseNoiseResult)
#     stdlistdenoise.append(round(chartRegionDeoise.std()))  # 标准差
#     msedenoiseResult = mse(chartRegion16,chartRegionDeoise)
#     msedenoiselist.append(msedenoiseResult)
#
#     S = np.linalg.norm(chartRegion16)
#     msesn = np.linalg.norm(chartRegionDeoise - np.mean(chartRegion16))
#     snresult = sn(S,msesn)
#     SNlist.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionDeoise))
#     PSNRlist.append(psnrresult)
#
# listall = [stdlist16,stdlist17,stdlistdenoise,meanlist16,SNlist,PSNRlist,msenoiselist,msedenoiselist]
# listcsv = pd.DataFrame(data=listall)
# listcsv.to_csv(os.path.join(csvsavepath,imagenumdenoise+'b.csv'))
#
#
# stdlist16 = []#标准差
# stdlist17 = []
# stdlistdenoise = []
# meanlist16 = []#均值
# meanlist17 = []
# msenoiselist = []
# msedenoiselist = []
# SNlist = []
# PSNRlist = []
#
# R16 = np.load(os.path.join(pathorigin,imagenumiso100+'g1.npy'))
# R17 = np.load(os.path.join(pathnoise,imagenumiso6400+'g1.npy'))
# Denoise = np.load(os.path.join(pathdenoise,imagenumdenoise+'g1.npy'))
#
# for i in range(24):
#     chartRegion16 = R16[list16[i][0]:list16[i][1],list16[i][2]:list16[i][3]]
#     chartRegion17 = R17[list17[i][0]:list17[i][1], list17[i][2]:list17[i][3]]
#     chartRegionDeoise = Denoise[list17[i][0]:list17[i][1], list17[i][2]:list17[i][3]]
#     #统一模块大小
#     list1= [chartRegion16.shape[0],chartRegion17.shape[0]]
#     list2 = [chartRegion16.shape[1],chartRegion17.shape[1]]
#     list1 = np.array(list1)
#     list2 = np.array(list2)
#     height = list1.min()
#     width = list2.min()
#     chartRegion16 = chartRegion16[0:height,0:width]
#     chartRegion17 = chartRegion17[0:height,0:width]
#     chartRegionDeoise = chartRegionDeoise[0:height, 0:width]
#
#     stdlist16.append(round(chartRegion16.std()))#标准差
#     meanlist16.append(round(np.mean(chartRegion16)))#均值
#
#     stdlist17.append(round(chartRegion17.std()))#标准差
#     mseNoiseResult = mse(chartRegion16,chartRegion17)
#     msenoiselist.append(mseNoiseResult)
#     stdlistdenoise.append(round(chartRegionDeoise.std()))  # 标准差
#     msedenoiseResult = mse(chartRegion16,chartRegionDeoise)
#     msedenoiselist.append(msedenoiseResult)
#
#     S = np.linalg.norm(chartRegion16)
#     msesn = np.linalg.norm(chartRegionDeoise - np.mean(chartRegion16))
#     snresult = sn(S,msesn)
#     SNlist.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionDeoise))
#     PSNRlist.append(psnrresult)
#
# listall = [stdlist16,stdlist17,stdlistdenoise,meanlist16,SNlist,PSNRlist,msenoiselist,msedenoiselist]
# listcsv = pd.DataFrame(data=listall)
# listcsv.to_csv(os.path.join(csvsavepath,imagenumdenoise+'g1.csv'))
#
# stdlist16 = []#标准差
# stdlist17 = []
# stdlistdenoise = []
# meanlist16 = []#均值
# meanlist17 = []
# msenoiselist = []
# msedenoiselist = []
# SNlist = []
# PSNRlist = []
#
# R16 = np.load(os.path.join(pathorigin,imagenumiso100+'g2.npy'))
# R17 = np.load(os.path.join(pathnoise,imagenumiso6400+'g2.npy'))
# Denoise = np.load(os.path.join(pathdenoise,imagenumdenoise+'g2.npy'))
#
# for i in range(24):
#     chartRegion16 = R16[list16[i][0]:list16[i][1],list16[i][2]:list16[i][3]]
#     chartRegion17 = R17[list17[i][0]:list17[i][1], list17[i][2]:list17[i][3]]
#     chartRegionDeoise = Denoise[list17[i][0]:list17[i][1], list17[i][2]:list17[i][3]]
#     #统一模块大小
#     list1= [chartRegion16.shape[0],chartRegion17.shape[0]]
#     list2 = [chartRegion16.shape[1],chartRegion17.shape[1]]
#     list1 = np.array(list1)
#     list2 = np.array(list2)
#     height = list1.min()
#     width = list2.min()
#     chartRegion16 = chartRegion16[0:height,0:width]
#     chartRegion17 = chartRegion17[0:height,0:width]
#     chartRegionDeoise = chartRegionDeoise[0:height, 0:width]
#
#     stdlist16.append(round(chartRegion16.std()))#标准差
#     meanlist16.append(round(np.mean(chartRegion16)))#均值
#
#     stdlist17.append(round(chartRegion17.std()))#标准差
#     mseNoiseResult = mse(chartRegion16,chartRegion17)
#     msenoiselist.append(mseNoiseResult)
#     stdlistdenoise.append(round(chartRegionDeoise.std()))  # 标准差
#     msedenoiseResult = mse(chartRegion16,chartRegionDeoise)
#     msedenoiselist.append(msedenoiseResult)
#
#     S = np.linalg.norm(chartRegion16)
#     msesn = np.linalg.norm(chartRegionDeoise - np.mean(chartRegion16))
#     snresult = sn(S,msesn)
#     SNlist.append(snresult)
#     psnrresult = snresult +PSNR(maxvalue,np.linalg.norm(chartRegionDeoise))
#     PSNRlist.append(psnrresult)
#
# listall = [stdlist16,stdlist17,stdlistdenoise,meanlist16,SNlist,PSNRlist,msenoiselist,msedenoiselist]
# listcsv = pd.DataFrame(data=listall)
# listcsv.to_csv(os.path.join(csvsavepath,imagenumdenoise+'g2.csv'))
