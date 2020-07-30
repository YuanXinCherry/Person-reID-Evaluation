import scipy.io
import numpy as np
import sys
import time
import T_MAC as t


#mAP_list, mmAP, method_name, line_type, line_color, line_width

method_name = ['ResNet-50', 'DenseNet-121', 'PCB', 'ResNet50_fc512', 'MLFN', 'HA-CNN', 'MobileNetV2_x1_0', 'MobileNetV2_x1_4', 
                'OSNet_x1_0', 'OSNet_x0_75', 'OSNet_x0_5', 'OSNet_x0_25', 'AlignedReID', 'BDB', 'BagTricks', 'AGW', 'ResNet50 + S-ReID',
                'DG-Net',  'FastReID(ResNet50)', 'MGN']
line_type = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.' ,'-', '--', '-.', '-', '--', '-.', '-', '--', '-.', '-', '--']
line_color = ['red', 'green', 'blue', 'tomato', 'sienna', 'darkorange', 'darkgoldenrod', 'gold', 'olive', 'yellow', 'lawngreen', 
                'palegreen', 'lime', 'cyan', 'dodgerblue', 'indigo', 'magenta', 'deeppink', 'crimson', 'lightpink']
line_width = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

num_method = len(method_name)

N = 100 
mAP_list = np.ones((num_method, N + 1))
mmAP = np.ones((num_method))

for i in range(num_method):
    result = '%s.mat'%method_name[i]
    result = scipy.io.loadmat(result)
    distance = result['distance']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]
    
    print(method_name[i])
    print(distance.shape)
    
    mAP_list[i], mmAP[i] = t.evaluate(distance,query_label,query_cam,gallery_label,gallery_cam,N)
    
    print(mAP_list[i])
    print(mmAP[i])
# Save to  mat for check
result = {'mAP_list': mAP_list, 'mmAP':mmAP}
scipy.io.savemat('CmAP_result.mat', result)

t.draw(mAP_list, mmAP, method_name, line_type, line_color, line_width, N)




