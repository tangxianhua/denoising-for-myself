import rawpy
import os
import glob
print('ARW2raw')
# print("请输入(空格隔开)文件夹绝对路径 文件存储绝对路径")
# totalinput = input()
# totalinput = totalinput.split(' ')
# if len(totalinput)!= 2:
#     print(r'参数错误! 命令格式为：inputpath outputpath')
# else:
#     inputpath = totalinput[0]
#     savepath = totalinput[1]
#     img_list = sorted(glob.glob(os.path.join(inputpath, '*.ARW')))
#     for z in range(len(img_list)):
#         basename = img_list[z].split('\\')[-1]
#         savename = basename.split('.')[0]
#         im = rawpy.imread(img_list[z])
#         raw = im.raw_image  # raw_image
#         raw.tofile(os.path.join(savepath, savename + '.raw'))
#         print('success')

print("请输入ARW单张文件绝对路径:（xxx.ARW） ")
inputpath = input()
print("请输入结果保存绝对路径:（xxx）")
savepath = input()
basename = inputpath.split('\\')[-1]
savename =basename.split('.')[0]
im = rawpy.imread(inputpath)
raw = im.raw_image#raw_image
raw.tofile(os.path.join(savepath,savename+'.raw'))
print('success')


