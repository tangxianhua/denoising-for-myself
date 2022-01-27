import numpy as np
import glob
import os
#raw2fourchannel
print('Bayer图像批量分解为4个单通道图像')
print("请输入(空格隔开)文件夹绝对路径 文件存储绝对路径 图像width 图像height")
totalinput = input()
totalinput = totalinput.split(' ')
if len(totalinput)!= 4:
    print(r'参数错误! 命令格式为：inputpath outputpath width height')
else:
    path = totalinput[0]
    savepath = totalinput[1]
    height = int(totalinput[2])
    width = int(totalinput[3])
    #print(height)
    img_list = sorted(glob.glob(os.path.join(path, '*.raw')))
    filelength = len(img_list)
    height1 = int(height/2)
    width1 = int(width/2)
    for z in range(filelength):
        imgfile = img_list[z]
        print("正在处理------------")
        print(imgfile)
        basenameraw = imgfile.split('\\')[-1]
        basename = basenameraw.split('.')[0]
        rawnpy = np.fromfile(imgfile, dtype='uint16')
        rawnpy = rawnpy.reshape(width, height)
        rawnpy = np.array(rawnpy)
        #R
        rnpy = np.zeros(height1*width1)
        ii = 0
        for i in range(0, width, 2):
            for j in range(0, height, 2):
                rnpy[ii] = rawnpy[i][j]
                ii += 1
        rnpy = rnpy.astype(np.uint16)
        rnpy = rnpy.reshape(width1, height1)
        rnpy.tofile(os.path.join(savepath,basename+'R'+'.raw'))
        #G1
        g1npy = np.zeros(height1*width1)
        jj = 0
        for i in range(0, width, 2):
            for j in range(1, height, 2):
                g1npy[jj] = rawnpy[i][j]
                jj += 1
        g1npy = g1npy.astype(np.uint16)
        g1npy = g1npy.reshape(width1, height1)
        g1npy.tofile(os.path.join(savepath,basename+'G1'+'.raw'))
        #G2
        g2npy = np.zeros(height1*width1)
        kk = 0
        for i in range(1, width, 2):
            for j in range(0, height, 2):
                g2npy[kk] = rawnpy[i][j]
                kk += 1
        g2npy = g2npy.astype(np.uint16)
        g2npy = g2npy.reshape(width1, height1)
        g2npy.tofile(os.path.join(savepath,basename+'G2'+'.raw'))
        #B
        bnpy = np.zeros(height1*width1)
        ll = 0
        for i in range(1, width, 2):
            for j in range(1, height, 2):
                bnpy[ll] = rawnpy[i][j]
                ll += 1
        bnpy = bnpy.astype(np.uint16)
        bnpy = bnpy.reshape(width1, height1)
        bnpy.tofile(os.path.join(savepath,basename+'B'+'.raw'))
