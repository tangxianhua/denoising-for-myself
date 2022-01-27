import numpy as np
import os
path = r'D:\Project\Python\Data\Data\sources\temp\6946\SFR\jbil'
imagename = 'jbillsb-'
num = '101520'
rnpy = np.load(os.path.join(path,imagename+num+'-r.npy'))
bnpy = np.load(os.path.join(path,imagename+num+'-b.npy'))
g1npy = np.load(os.path.join(path,imagename+num+'-g1.npy'))
g2npy = np.load(os.path.join(path,imagename+num+'-g2.npy'))

# rnpy = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\jgt\tvm\tvm01-r.npy')
# bnpy = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\jgt\tvm\tvm01-b.npy')
# g1npy = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\jgt\guide\g01-g1.npy')
# g2npy = np.load(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\jgt\guide\g01-g2.npy')

rnpy1 = rnpy.flatten()
bnpy1 = bnpy.flatten()
g1npy1 = g1npy.flatten()
g2npy1 = g2npy.flatten()
bayernpy = np.zeros(160000)#24337152
ii = 0
jj = 0
for i in range(0,160000,2):#24337152
    if ((int((i/400)))%2 ) == 0:#6048
        bayernpy[i] = rnpy1[ii]
        bayernpy[i+1] = g1npy1[ii]
        ii+=1

    if ((int((i /400 ))) % 2) == 1:#6048
        bayernpy[i] = g2npy1[jj]
        bayernpy[i+1] = bnpy1[jj]
        jj+=1
print(i)
height=400#6048
width=400#4024
bayernpy = bayernpy.reshape(width, height)
bayernpy = bayernpy.astype(np.uint16)
np.save(os.path.join(path,imagename+num+'.npy'),bayernpy)
bayernpy.tofile(os.path.join(path,imagename+num+'.raw'))
# np.save(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\jgt\diff\diff-01.npy',bayernpy)
# bayernpy.tofile(r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\30vs31vs32\jgt\diff\diff-01.raw')









# rbpath = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\samegain\sony6400\plottest\tvm'
# gpath = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\samegain\sony6400\plottest\guide'
# savepath = r'D:\Project\Python\Data\Data\sources\jbilateral\moreSampleTest\sfr\samegain\sony6400\plottest\diff'
# count = 0
# for j in range(1,5):
#     jname = str(j)
#     for g in range(1,4):
#         gname = str(g)
#         for t in range(1,4):
#             tname = str(t)
#             rbname = 'j'+jname+'g'+gname+'t'+tname+'-'
#             rnpy = np.load(os.path.join(rbpath, rbname + 'r.npy'))
#             bnpy = np.load(os.path.join(rbpath, rbname + 'b.npy'))
#             gggname = 'j0'+jname+'-'+'g0'+gname+'-'
#             g1npy = np.load(os.path.join(gpath, gggname + 'g1.npy'))
#             g2npy = np.load(os.path.join(gpath, gggname + 'g2.npy'))
#             savename = rbname+gggname
#             print(savename)
#             rnpy1 = rnpy.flatten()
#             bnpy1 = bnpy.flatten()
#             g1npy1 = g1npy.flatten()
#             g2npy1 = g2npy.flatten()
#             bayernpy = np.zeros(24337152)
#             ii = 0
#             jj = 0
#             for i in range(0, 24337152, 2):
#                 if ((int((i / 6048))) % 2) == 0:
#                     bayernpy[i] = rnpy1[ii]
#                     bayernpy[i + 1] = g1npy1[ii]
#                     ii += 1
#                 if ((int((i / 6048))) % 2) == 1:
#                     bayernpy[i] = g2npy1[jj]
#                     bayernpy[i + 1] = bnpy1[jj]
#                     jj += 1
#             height = 6048
#             width = 4024
#             bayernpy = bayernpy.reshape(width, height)
#             bayernpy = bayernpy.astype(np.uint16)
#             np.save(os.path.join(savepath,savename+'.npy'),bayernpy)
#             bayernpy.tofile(os.path.join(savepath,savename+'.raw'))
#             print(count)
#             count+=1