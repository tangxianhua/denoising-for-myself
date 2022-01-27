import numpy as np
import glob
import os
import time
#sum-mean
print('sum-mean(文件夹下所有文件相加后求均值文件)')
print("请输入(空格隔开)文件夹绝对路径 文件存储绝对路径 图像width 图像height offset-num1 offset-num2 offset-num3 offset-num4")
totalinput = input()
totalinput = totalinput.split(' ')
if len(totalinput)!= 8:
    print(r'参数错误! 命令格式为：inputpath outputpath width height offset[4]')
else:
    path = totalinput[0]
    savepath = totalinput[1]
    height = int(totalinput[2])
    width = int(totalinput[3])
    offset = [int(n) for n in totalinput[4:]]
    #print(offset)
    img_list = sorted(glob.glob(os.path.join(path, '*.raw')))
    filelength = len(img_list)
    allnpy = np.zeros((width, height))
    for z in range(filelength):
        imgfile = img_list[z]
        print("正在处理------------")
        print(imgfile)
        basenameraw = imgfile.split('\\')[-1]
        basename = basenameraw.split('.')[0]
        rawnpy = np.fromfile(imgfile, dtype='uint16')
        rawnpy = rawnpy.reshape(width, height)
        rawnpy = np.array(rawnpy)
        if z == (filelength - 1):
            for i in range(0, width, 2):
                for j in range(0, height, 2):
                    rawnpy[i][j] -= offset[0]
                    allnpy[i][j] += rawnpy[i][j]
                    allnpy[i][j] /= filelength
                    allnpy[i][j] += offset[0]
            for i in range(0, width, 2):
                for j in range(1, height, 2):
                    rawnpy[i][j] -= offset[1]
                    allnpy[i][j] += rawnpy[i][j]
                    allnpy[i][j] /= filelength
                    allnpy[i][j] += offset[1]
            for i in range(1, width, 2):
                for j in range(0, height, 2):
                    rawnpy[i][j] -= offset[2]
                    allnpy[i][j] += rawnpy[i][j]
                    allnpy[i][j] /= filelength
                    allnpy[i][j] += offset[2]
            for i in range(1, width, 2):
                for j in range(1, height, 2):
                    rawnpy[i][j] -= offset[3]
                    allnpy[i][j] += rawnpy[i][j]
                    allnpy[i][j] /= filelength
                    allnpy[i][j] += offset[3]
        else:
            for i in range(0, width, 2):
                for j in range(0, height, 2):
                    rawnpy[i][j] -= offset[0]#r2
                    allnpy[i][j] += rawnpy[i][j]
            for i in range(0, width, 2):
                for j in range(1, height, 2):
                    rawnpy[i][j] -= offset[1]#g1
                    allnpy[i][j] += rawnpy[i][j]
            for i in range(1, width, 2):
                for j in range(0, height, 2):
                    rawnpy[i][j] -= offset[2]#g2
                    allnpy[i][j] += rawnpy[i][j]
            for i in range(1, width, 2):
                for j in range(1, height, 2):
                    rawnpy[i][j] -= offset[3]#b
                    allnpy[i][j] += rawnpy[i][j]
        # for i in range(width):
        #     for j in range(height):
        #         allnpy[i][j] += rawnpy[i][j]
    # for i in range(0, width, 2):
    #     for j in range(0, height, 2):
    #         allnpy[i][j] /= filelength
    #         allnpy[i][j] += offset[0]#r
    # for i in range(0, width, 2):
    #     for j in range(1, height, 2):
    #         allnpy[i][j] /= filelength
    #         allnpy[i][j] += offset[1]#g1
    # for i in range(1, width, 2):
    #     for j in range(0, height, 2):
    #         allnpy[i][j] += offset[2]#g2
    # for i in range(1, width, 2):
    #     for j in range(1, height, 2):
    #         allnpy[i][j] /= filelength
    #         allnpy[i][j] += offset[3]#b
    timename = int(time.time())
    savename = os.path.join(savepath,str(timename)+'mean'+'.raw')
    allnpy = allnpy.astype(np.uint16)
    allnpy.tofile(savename)