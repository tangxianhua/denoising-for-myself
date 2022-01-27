import numpy as np
import pandas as pd
import os
#cal-snr
print('calculate-snr')
print("请输入(空格隔开)文件1绝对路径 文件2绝对路径 csv文件存储绝对路径 图像width 图像height 计算窗口左上角y坐标 右下角y坐标 左上角x坐标 左下角x坐标")
totalinput = input()
totalinput = totalinput.split(' ')
if len(totalinput)!= 9:
    print(r'参数错误! 命令格式为：input1path input2path csvoutputpath width height y1 y2 x3 x4')
else:
    path1 = totalinput[0]
    path2 = totalinput[1]
    savepath = totalinput[2]
    height = int(totalinput[3])
    width = int(totalinput[4])
    win = [int(n) for n in totalinput[5:]]
    npy1 = np.fromfile(path1, dtype='uint8')
    npy1 = npy1.reshape(width, height)
    npy1 = np.array(npy1)
    npy2 = np.fromfile(path2, dtype='uint8')
    npy2 = npy2.reshape(width, height)
    npy2 = np.array(npy2)
    list1 = []
    Region1 = npy1[win[0]:win[1],win[2]:win[3]]
    Region2 = npy2[win[0]:win[1],win[2]:win[3]]
    S = np.linalg.norm(Region1)
    MSE = np.linalg.norm(Region1-np.mean(Region2))
    SN = 20*(np.log10(S/MSE))
    list1.append(SN)

    listall = list1
    listcsv = pd.DataFrame(data=listall)
    savename = os.path.join(savepath,'result.csv')
    listcsv.to_csv(savename)