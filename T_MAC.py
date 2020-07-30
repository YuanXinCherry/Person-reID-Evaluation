import scipy.io
import sys
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 
import numpy as np
from numpy.linalg import norm
from matplotlib.patches import ConnectionPatch
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler


def norm_distance(distance):

    min_max_scaler=MinMaxScaler()
    X_train_minmax=min_max_scaler.fit_transform(distance.T)
    distance = X_train_minmax.T
     
    return  distance


#######################################################################
# Evaluate
# dismat:distance-matrix;ql:query-labels;qc:query-cameras;gl:gallery-labels;gc:gallery-cameras;
def evaluate(dismat, ql, qc, gl, gc, n_bin=100): 
    
    distance = norm_distance(dismat)
    n_q, n_g = distance.shape
    
    junk_index1 = np.argwhere(gl==-1)
    distance[:,junk_index1] = 1
    gl = np.tile(gl,(n_q,1))
    
    for q_junk_idx in range(n_q):
        q_index = np.argwhere(gl[q_junk_idx,:]==ql[q_junk_idx])
        c_index = np.argwhere(gc==qc[q_junk_idx])
        junk_index = np.intersect1d(q_index, c_index)
        distance[q_junk_idx, junk_index] = 1
        gl[q_junk_idx, junk_index] = -1
        
    distance_idx = np.argsort(distance, axis=1)
    matches = np.zeros((n_q, n_g))
    
    for i in range(n_q):
        distance[i,:] = distance[i,score_idx[i,:]]
        gl[i,:] = gl[i, distance_idx[i,:]]
        matches[i,:] = (gl[i,:] == ql[i]).astype(np.int32)
    
      
    CmAP = np.zeros((n_bin + 1)) 
    cur_gt_count = np.zeros(n_bin + 1) 
    
    threshold_bin = np.arange(0, 1 + 1.0/n_bin, 1.0/n_bin)
    threshold_bin = np.tile(threshold_bin, (n_g,1))
    threshold_bin = threshold_bin.T
    
    n_sample = np.arange(1,n_g+1,1)
    n_sample = np.tile(n_sample,(n_bin + 1,1))
   
    # find good index for every query
    for q_idx in range(n_q):
        view_bar(q_idx, n_q)
        
        q_distance = np.tile(distance[q_idx], (n_bin + 1,1)) 
        q_distance = q_distance - threshold_bin 
        q_mask = np.where(q_distance <= 0, 1, 0)
        q_gt = np.tile(matches[q_idx], (n_bin + 1,1))
        q_gt_mask = q_mask*q_gt
        q_gt_mask_bk = np.copy(q_gt_mask)
        n_q_gt = q_gt_mask.sum(axis=1)
        
        for i in range(n_g):
            cur_gt_count = q_gt_mask[:,i] + cur_gt_count
            q_gt_mask[:,i] = cur_gt_count
        cur_gt_count = 0 
        
        q_gt_mask = q_gt_mask*q_gt_mask_bk
        q_ap = q_gt_mask*1.0/n_sample
        q_ap = q_ap.sum(axis=1)
        
        n_q_gt = np.where(n_q_gt==0,1,n_q_gt)
        q_ap = q_ap/n_q_gt
        
        CmAP = CmAP + q_ap
        
    CmAP = CmAP/n_q
    mmAP = np.mean(CmAP)   
    return CmAP, mmAP

def view_bar(num, total):
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%,%d/%d' % ("="*rate_num, " "*(100-rate_num), rate_num, num,total )
    sys.stdout.write(r)
    sys.stdout.flush()


def draw(mAP_list, mmAP, method_name, line_type, line_color, line_width, N):
    num = len(method_name) 
    x = np.arange(0.0, 1.0 + 1.0/N, 1.0/N)
    
    
    plt.figure(figsize=(16,6),dpi=98)
    p1 = plt.subplot(121)
    p2 = plt.subplot(122)
    
    for i in range(num):
        reid_idx = np.argmax(mAP_list[i]) 
        retr_idx = np.argwhere(mAP_list[i] == mAP_list[i][-1])
        curve_label = ''.join(method_name[i]) + ' (mAP='+ '%.2f%%' % ( mAP_list[i][-1]* 100) +', λ1='+ '%.2f' %(x[reid_idx])+', λ2='+'%.2f' %(x[retr_idx[0]])+')'
        p1.plot(x, mAP_list[i], label = curve_label, linestyle = line_type[i], color = line_color[i], linewidth = line_width[i])
        p2.plot(x, mAP_list[i], label = curve_label, linestyle = line_type[i], color = line_color[i], linewidth = line_width[i])

    #p1
    p1.axis([0.0,1.0,0.0,1.0])
    p1.set_xlabel('Threshold', fontsize=14, color='black')
    p1.set_ylabel('mAP', fontsize=14, color='black')
    p1.set_title('T-MAC') 
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    p1.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, fontsize = 8)
    
    #p2
    p2.axis([0,0.4,0.5,1.0])
    p2.set_xlabel('Threshold', fontsize=14, color='black')
    p2.set_ylabel('mAP', fontsize=14, color='black')
    p2.set_title('T-MAC') 
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    p2.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0, fontsize = 8)

    # plot the box
    tx0 = 0
    tx1 = 0.4
    ty0 = 0.5
    ty1 = 1
    sx = [tx0,tx1,tx1,tx0,tx0]
    sy = [ty0,ty0,ty1,ty1,ty0]
    p1.plot(sx,sy,"purple")

    plt.show()
    plt.savefig('T-MAC.jpg')
    plt.savefig('T-MAC.pdf')
    

