import sys, time
import scipy.io
import sys
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from numpy import linspace
from scipy import integrate
from numpy.linalg import norm
from matplotlib.patches import ConnectionPatch
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler

class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self, s):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        print(s)
        progress = self.width * self.count / self.total
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('#' * int(progress) + '-' * int(self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


def norm_distance(distance):
    distance = (distance - distance.min()) / (distance.max() - distance.min())

    return distance

###############################################################################################
# Evaluate closed-world
# dismat:distance-matrix;ql:query-labels;qc:query-cameras;gl:gallery-labels;gc:gallery-cameras;
def evaluate_closed(dismat, ql, qc, gl, gc, n_bin=100):

    # If the distance matrix is not normalized, normalization is required.
    # distance = norm_distance(dismat)
    distance = dismat
    n_q, n_g = distance.shape
    print(n_q, n_g)
    
    # Only part of body is detected and the distance values are set with 1.
    junk_index1 = np.argwhere(gl == -1)
    distance[:, junk_index1] = 1
    gl = np.tile(gl, (n_q, 1))

    # The images of the same identity in same cameras, distance values are set with 1
    for q_junk_idx in range(n_q):
        q_index = np.argwhere(gl[q_junk_idx, :] == ql[q_junk_idx])
        c_index = np.argwhere(gc == qc[q_junk_idx])
        junk_index = np.intersect1d(q_index, c_index)
        distance[q_junk_idx, junk_index] = 1
        gl[q_junk_idx, junk_index] = -1

    # Sort the distance matrix from smallest to largest
    distance_idx = np.argsort(distance, axis=1)
    matches = np.zeros((n_q, n_g))

    # Sort the matches and gl by distance from smallest to largest.
    for i in range(n_q):
        distance[i, :] = distance[i, distance_idx[i, :]]
        gl[i, :] = gl[i, distance_idx[i, :]]
        matches[i, :] = (gl[i, :] == ql[i]).astype(np.int32)

    
    num_valid_q = 0.                   # number of valid query
    all_cmc = []                       # all cmc list
    mRP = np.zeros((n_bin + 1))        # mRP list
    mReP = np.zeros((n_bin + 1))       # mReP list
    mVP = np.zeros((n_bin + 1))        # mVP list
    cur_gt_count = np.zeros(n_bin + 1) # number of current gt

    # threshold_bin matrix
    threshold_bin = np.arange(0, 1 + 1.0 / n_bin, 1.0 / n_bin)
    threshold_bin = np.tile(threshold_bin, (n_g, 1))
    threshold_bin = threshold_bin.T

    # sample matrix
    n_sample = np.arange(1, n_g + 1, 1)
    n_sample = np.tile(n_sample, (n_bin + 1, 1))

    # ProgressBar object 
    bar = ProgressBar(total = n_q)
    for q_idx in range(n_q):
        # Setting the progress bar
        bar.move()
        bar.log(str(q_idx + 1))
        time.sleep(0.001)

        # Get q_mask according to distance
        q_distance = np.tile(distance[q_idx], (n_bin + 1, 1))
        q_distance = q_distance - threshold_bin
        q_mask = np.where(q_distance <= 0, 1, 0)
        q_gt = np.tile(matches[q_idx], (n_bin + 1, 1))
        q_gt_mask = q_mask * q_gt
        q_gt_mask_bk = q_gt_mask.copy()

        # Calculate the number of gt and positive (returned results) under different thresholds
        n_q_gt = 0
        n_q_gt = q_gt_mask.sum(axis=1)    # number of gt under different thresholds
        n_q_positive = q_mask.sum(axis=1) # number of positive under different thresholds

        # all cmc
        raw_cmc = matches[q_idx]
        if not np.any(raw_cmc):
            continue
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:10])
        num_valid_q += 1.

        # all ap
        for i in range(n_g):
            cur_gt_count = q_gt_mask[:, i] + cur_gt_count
            q_gt_mask[:, i] = cur_gt_count
        cur_gt_count = 0
        q_gt_mask = q_gt_mask * q_gt_mask_bk
        q_ap = q_gt_mask * 1.0 / n_sample
        q_ap = q_ap.sum(axis=1)

        # Record the last matches position
        matches_idx = np.argwhere(matches[q_idx] == 1)
        n_q_count = matches_idx[-1] + 1

        # After returning all gt, all subsequent positives are set with the number of gt.
        n_q_positive = np.where(n_q_positive > n_q_count, n_q_count, n_q_positive)
        print(n_q_positive)
      
        # Calculates ap and returns the result of setting -1 to 0.
        n_q_gt = np.where(n_q_gt == 0, -1, n_q_gt)
        q_ap = q_ap / n_q_gt
        n_q_gt = np.where(n_q_gt == -1, 0, n_q_gt)

        # VP and ReP
        if (n_q_count - n_q_gt[-1]) >= 0:
            q_vp = (n_q_gt * 1.0) / (n_q_positive - n_q_gt + n_q_gt[-1])
            mVP = mVP + q_vp
            mReP = mReP + np.sqrt(q_ap * q_vp)

        # mRP
        mRP = mRP + q_ap

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q           # cmc                                
    mRP = mRP / n_q                                  # mRP
    mVP = mVP / n_q                                  # mVP
    mReP = mReP / n_q                                # mReP

    return all_cmc, mRP, mVP, mReP

#######################################################################
# Evaluate open-set
# dismat:distance-matrix;ql:query-labels;qc:query-cameras;gl:gallery-labels;gc:gallery-cameras;
def evaluate_open(dismat, ql, qc, gl, gc, B=3000, n_bin=100): 
    
    # If the distance matrix is not normalized, normalization is required.
    # distance = norm_distance(dismat)
    distance = dismat
    n_q, n_g = distance.shape
    print(n_q, n_g)
    
    # Only part of body is detected and the distance values are set with 1.
    junk_index1 = np.argwhere(gl == -1)
    distance[:, junk_index1] = 1
    gl = np.tile(gl, (n_q, 1))

    # The images of the same identity in same cameras, distance values are set with 1
    for q_junk_idx in range(n_q):
        q_index = np.argwhere(gl[q_junk_idx, :] == ql[q_junk_idx])
        c_index = np.argwhere(gc == qc[q_junk_idx])
        junk_index = np.intersect1d(q_index, c_index)
        distance[q_junk_idx, junk_index] = 1
        gl[q_junk_idx, junk_index] = -1

    # Sort the distance matrix from smallest to largest
    distance_idx = np.argsort(distance, axis=1)
    matches = np.zeros((n_q, n_g))

    # Sort the matches and gl by distance from smallest to largest.
    for i in range(n_q):
        distance[i, :] = distance[i, distance_idx[i, :]]
        gl[i, :] = gl[i, distance_idx[i, :]]
        matches[i, :] = (gl[i, :] == ql[i]).astype(np.int32)
      
    
    mFR = np.zeros((n_bin + 1))             # mFR list
    
    # threshold_bin matrix
    threshold_bin = np.arange(0, 1 + 1.0/n_bin, 1.0/n_bin)
    threshold_bin = np.tile(threshold_bin, (n_g,1))
    threshold_bin = threshold_bin.T
    
    # sample matrix
    n_sample = np.arange(1,n_g+1,1)
    n_sample = np.tile(n_sample,(n_bin + 1,1))
   
    # ProgressBar object 
    bar = ProgressBar(total = n_q)
    for q_idx in range(n_q):
        # Setting the progress bar
        bar.move()
        bar.log(str(q_idx + 1))
        time.sleep(0.001)
        
        # Get q_mask according to distance
        q_distance = np.tile(distance[q_idx], (n_bin + 1,1)) 
        q_distance = q_distance - threshold_bin 
        q_mask = np.where(q_distance <= 0, 1, 0)
        n_q_positive = q_mask.sum(axis=1)

        # Record the number of results returned under different thresholds
        n_q_positive = np.where(n_q_positive > B, B, n_q_positive)
        print(n_q_positive)
  
        # mFR
        mFR = mFR + n_q_positive*1.0/B

    # mFR 
    mFR = mFR / n_q
    return mFR

# print rank@1, mAP, mVP_{max}, mReP_{max}, MREP, MFR
def print_GOM(method_name, CMC_list, mRP_list, mVP_list, mReP_list, mFR_list, N=100):
    num = len(method_name)
    x = np.arange(0.0, 1.0 + 1.0 / N, 1.0 / N)
    for i in range(num):
        print(method_name[i])
        print('Existing')
        rank1 = CMC_list[i][0]
        mAP = mRP_list[i][-1]
        print('Rank@1:%f mAP:%f '%(rank1, mAP))
        
        print('Proposed')
        mVP_max = np.max(mVP_list[i])
        mReP_max = np.max(mReP_list[i])
        #MREP = mReP_list[i][0:N:1]
        #MFR = mFR_list[i][0:N:1]

        #MREP = np.sum(MREP)/len(MREP)
        #MFR = np.sum(MFR)/len(MFR)
        mReP = np.array(mReP_list[i])
        mFR = np.array(mFR_list[i])
        MREP = integrate.trapz(mReP, x)
        MFR = integrate.trapz(mFR, x)
        maxidx = np.argmax(mReP_list[i])
        print('mVP_max:%f mReP_max:%f tau_max:%f MREP:%f MFR:%f'%(mVP_max, mReP_max, maxidx*1.0/N, MREP, MFR))                                


def draw(mReP_list, mFR_list, method_name, line_type, line_color, line_width, location=(0.01, 0.08), N=100):
    plt.figure(figsize=(10,12), dpi=80)
    a13 = plt.subplot(211)
    a24 = plt.subplot(212)

    num = len(method_name)
    x = np.arange(0.0, 1.0 + 1.0 / N, 1.0 / N)
    for i in range(num):
        a13.plot(x, mReP_list[i], linestyle = line_type[i], color = line_color[i], linewidth = line_width[i])

        down_label = ''.join(method_name[i]) 
        a24.plot(x, mFR_list[i], label = down_label, linestyle = line_type[i], color = line_color[i], linewidth = line_width[i])

    #axx.axis('off')
    #plt.subplots_adjust(right=0.78)
    plt.subplots_adjust(wspace=0, hspace=0.1)

    a13.set_xlim([0,1])
    a13.tick_params(labelsize=30)
    a13.set_yticks([0.0,0.2,0.4,0.6,0.8])

    a13.set_xlabel('', fontsize=30, color='black')
    a13.set_ylabel('mReP', fontsize=30, color='black')
    #a13.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=0, ncol=1, mode="expand", borderaxespad=0.)
    #a13.legend(bbox_to_anchor=(0., -1.02, 1., -.102),loc=0, ncol=1, mode="expand", borderaxespad=0.)
    #a24.legend(bbox_to_anchor=(1.02, 0, 0, 0), loc=3, borderaxespad=0, fontsize = 10)

    a24.set_xlim([0,1])
    a24.tick_params(labelsize=30)
    a24.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])

    a24.set_xlabel('Threshold', fontsize=30, color='black')
    a24.set_ylabel('mFR', fontsize=30, color='black')
    a24.legend(loc=location, fontsize=26, labelspacing=0.2)
    #plt.subplots_adjust(wspace=0, hspace=0)  
    #fig.legend(bbox_to_anchor=(0.5, 0.6), loc=3, borderaxespad=0, fontsize = 10)
    plt.savefig('mReP-mFR.jpg')
    plt.savefig('mReP-mFR.pdf')
    plt.show()


######################################################################
# test
'''
result = scipy.io.loadmat('100_AGW.mat')
distance = result['distmat']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

print(distance.shape)
print(query_cam.shape)
print(query_label.shape)
print(gallery_cam.shape)
print(gallery_label.shape)

#cmc, mRP_list, mVP_list, mReP_list = evaluate_closed(distance, query_label, query_cam, gallery_label, gallery_cam, 100)
mFR_list = evaluate_open(distance, query_label, query_cam, gallery_label, gallery_cam, 100)
# mAP_list = np.arange(0.0, 1.0, 1.0/100)
# mmAP = 0.21
# test_draw(mAP_list, mVP_list, mReP_list, 'AGW', '-', 'red', 2, 100)

# print(cmc[0])
# print(mRP_list[-1])
# print(mRP_list)
# print(mVP_list)
# print(mReP_list)

print(mFR_list)
'''
# print(max_return_num)

# Save to  mat for check
# result = {'mAP_list': mAP_list, 'mmAP':mmAP, 'mReP_list':mReP_list}
# scipy.io.savemat('./mReP/mReP-AGW-market.mat', result)
