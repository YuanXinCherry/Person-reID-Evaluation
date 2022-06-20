import scipy.io
import numpy as np
import sys
import time
import GOM as gom


#CMC_list, mRP_list, mVP_list, mReP_list, mFR_list, method_name, line_type, line_color, line_width

# The example of multiple methods 
method_name = ['ResNet-50', 'DenseNet-121', 'MLFN', 'HA-CNN', 'MobileNetV2', 'OSNet', 'AlignedReID', 'DG-Net', 'BDB', 'BagTricks', 'FastReID', 'AGW']
line_type = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.' ,'-', '--', '-.' ]
line_color = ['red', 'green', 'blue', 'tomato', 'sienna', 'darkorange', 'darkgoldenrod', 'gold', 'olive', 'yellow', 'lawngreen', 'palegreen']
line_width = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

# The example of one methods
#method_name = ['ResNet-50']
#line_type = ['-']
#line_color = ['red']
#line_width = [2]

num_method = len(method_name)

# CMC_list, mRP_list, mVP_list, mReP_list, mFR_list
N = 100
B = 3000
# legend loc
location = (0.01, 0.08)

CMC_list = np.ones((num_method, 10))
mRP_list = np.ones((num_method, N + 1))
mVP_list = np.ones((num_method, N + 1))
mReP_list = np.ones((num_method, N + 1))
mFR_list = np.ones((num_method, N + 1))

for i in range(num_method):
    print(method_name[i])
    # closed-world
    closed_result = '3368_%s.mat'%method_name[i]
    closed_result = scipy.io.loadmat(closed_result)
    closed_distance = closed_result['distmat']
    closed_query_cam = closed_result['query_cam'][0]
    closed_query_label = closed_result['query_label'][0]
    closed_gallery_cam = closed_result['gallery_cam'][0]
    closed_gallery_label = closed_result['gallery_label'][0]
    print(closed_distance.shape)
    CMC_list[i], mRP_list[i], mVP_list[i], mReP_list[i] = gom.evaluate_closed(closed_distance, closed_query_label, closed_query_cam, closed_gallery_label, closed_gallery_cam,N)
    
    # open-set 
    open_result = '100_%s.mat'%method_name[i]
    open_result = scipy.io.loadmat(open_result)
    open_distance = open_result['distmat']
    open_query_cam = open_result['query_cam'][0]
    open_query_label = open_result['query_label'][0]
    open_gallery_cam = open_result['gallery_cam'][0]
    open_gallery_label = open_result['gallery_label'][0]
    print(open_distance.shape)
    mFR_list[i] = gom.evaluate_open(open_distance, open_query_label, open_query_cam, open_gallery_label, open_gallery_cam, B, N)

# print rank@1, mAP, mVP_{max}, mReP_{max}, MREP, MFR
gom.print_GOM(method_name, CMC_list, mRP_list, mVP_list, mReP_list, mFR_list, N)

# Save to  mat for check
GOM_result = {'CMC_list': CMC_list, 'mRP_list': mRP_list, 'mVP_list':mVP_list, 'mReP_list':mReP_list, 'mFR_list':mFR_list}
scipy.io.savemat('GOM-market1501.mat', GOM_result)

gom.draw(mReP_list, mFR_list, method_name, line_type, line_color, line_width, location, N)




