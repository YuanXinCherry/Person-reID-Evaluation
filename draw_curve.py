# -*- coding: utf-8 -*

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
import numpy as np
from xlrd import open_workbook





#x = np.arange(0.0, 1.0, 0.001)#x轴上的点，0到1之间以0.001为间隔  
#mAP = evaluate(qf,ql,qc,gf,gl,gc)
x=[]
mAP_resnet50=[]
mAP_resnet50_fc512=[]
mAP_mlfn=[]
mAP_hacnn=[]
mAP_mobilenetv2_x1_0=[]
mAP_mobilenetv2_x1_4=[]
mAP_osnet_x1_0=[]
mAP_osnet_x0_75=[]
mAP_osnet_x0_5=[]
mAP_osnet_x0_25=[]

wb = open_workbook('all-method.xlsx')
for s in wb.sheets():
    print('Sheet:',s.name)
    for row in range(s.nrows):
        print('the row is:',row)
        values = []
        for col in range(s.ncols):
            values.append(s.cell(row,col).value)
        print(values)
        x.append(values[0])
        mAP_resnet50.append(values[1])
        mAP_resnet50_fc512.append(values[2])
        mAP_mlfn.append(values[3])
        mAP_hacnn.append(values[4])
        mAP_mobilenetv2_x1_0.append(values[5])
        mAP_mobilenetv2_x1_4.append(values[6])
        mAP_osnet_x1_0.append(values[7])
        mAP_osnet_x0_75.append(values[8])
        mAP_osnet_x0_5.append(values[9])
        mAP_osnet_x0_25.append(values[10])


with plt.style.context(['ieee']):
    fig, ax = plt.subplots()
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])  # 设置x刻度
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])  # 设置y刻度
    ax.plot(x, mAP_resnet50,label='resnet50',c='b',linestyle='-',linewidth=1, marker='o',markersize=2)
    ax.plot(x, mAP_resnet50_fc512,label='resnet50_fc512',c='r',linestyle='-',linewidth=1, marker='p',markersize=2)
    ax.plot(x, mAP_mlfn, label='mlfn',c='c',linestyle='-',linewidth=1, marker='D',markersize=2)
    ax.plot(x, mAP_hacnn, label='hacnn',c='m',linestyle='-',linewidth=1, marker='v',markersize=2)
    ax.plot(x, mAP_mobilenetv2_x1_0, label='mobilenetv2_x1_0',c='g',linestyle='-', linewidth=1, marker='s',markersize=2)
    ax.plot(x, mAP_mobilenetv2_x1_4, label='mobilenetv2_x1_4',c='y',linestyle='-',linewidth=1, marker='*',markersize=2)
    ax.plot(x, mAP_osnet_x1_0, label='osnet_x1_0',c='k',linestyle='-', linewidth=1,marker='H',markersize=2)
    ax.plot(x, mAP_osnet_x0_75, label='osnet_x0_75',c='lime',linestyle='-',linewidth=1, marker='x',markersize=2)
    ax.plot(x, mAP_osnet_x0_5, label='osnet_x0_5',c='dodgerblue',linestyle='-', linewidth=1,marker='d',markersize=2)
    ax.plot(x, mAP_osnet_x0_25, label='osnet_x0_25',c='pink',linestyle='-', linewidth=1,marker='^',markersize=2)
    
    #ax.legend(['resnet50','resnet50_fc512','mlfn','hacnn','mobilenetv2_x1_0','mobilenetv2_x1_4','osnet_x1_0','osnet_x0_75','osnet_x0_5','osnet_x0_25'])
    ax.legend(title='Method',loc='lower right',fontsize='xx-small')
    ax.set(xlabel='Threshold')
    ax.set(ylabel='mAP')
    ax.autoscale(tight=True)
    
    fig.show()
    fig.savefig('fig1.pdf')
    fig.savefig('fig1.jpg')
