# -*- coding: utf-8 -*

import scipy.io
import torch
import numpy as np
import sys
import time
#import os

N=1000 #阈值间隔
good_size=0 #正确匹配的样本数

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):      #参数qf:query-features;ql:query-labels;qc:query-cameras;gf:gallry-features;gl:gallery-labels;gc:gallery-cameras;
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query) #矩阵相乘
    score = score.squeeze(1).cpu()
    score = score.numpy()
    #上一步计算相似度分数
    #print(score[0])
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    
    # The images of the same identity in different cameras
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # Only part of body is detected. 
    junk_index1 = np.argwhere(gl==-1) #错误检测的图像，主要是包含一些人的部件。
    # The images of the same identity in same cameras
    junk_index2 = np.intersect1d(query_index, camera_index)#相同的人在同一摄像头下，按照reid的定义，我们不需要检索这一类图像
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    #输出相似度分数
    #print('%f\n' %(score[index[0]]))
    ap = compute_mAP(index, good_index, junk_index, score)
    
    return ap
    
def compute_mAP(index, good_index, junk_index, score):
    global good_size #需要修改全局变量
    ap = torch.IntTensor(N+1).zero_()
    if good_index.size==0:   # if empty
        good_size=-1
        return ap

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask)
    rows_good = rows_good.flatten()
    
    #mAP-threshold关系
    j=0
    k=0
    t=0.0
    for i in range(ngood):
        if rows_good[i]!=0:
            j = j + 1
            idx = index[i]
            while score[idx] > k*1.0/N and k<=N:
                k = k + 1
                #print(k)
            t = t + (i+1)*1.0/(rows_good[i]+1)
            ap[k-1:] = t/j

    return ap

# 循环时显示进度条
# total 代表循环总数 ，num为当前循环数

def view_bar(num, total):
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%,%d/%d' % ("="*rate_num, " "*(100-rate_num), rate_num, num,total )
    sys.stdout.write(r)
    sys.stdout.flush()
'''
[=========                           ]9%,9/20
'''
    
######################################################################
result = scipy.io.loadmat('pytorch_result_Resnet50.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)

mAP = torch.IntTensor(N+1).zero_() #my
mAP = mAP.float()



#print(query_label)

for i in range(len(query_label)):
    time.sleep(1)
    view_bar(i , len(query_label))
    ap_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if good_size==-1:
        continue
    mAP = mAP + ap_tmp

mAP = mAP/len(query_label) #compute_mAP   
print('\n')
for i in range(N+1):
    print('mAP:%f'%mAP[i])

