import scipy.io
import numpy as np
import sys
import time
import T_MAC as t


#mAP_list, mmAP, method_name, line_type, line_color, line_width

method_name = ['Resnet50', 'Densenet121', 'PCB', 'resnet50_fc512', 'mlfn', 'hacnn', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 
                'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'alignedreid', 'bfenet', 'BagTricks', 'AGW', 'SReID',
                'DGNet',  'fastreid', 'MGN']
line_type = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.' ,'-', '--', '-.', '-', '--', '-.', '-', '--', '-.', '-', '--']
line_color = ['red', 'green', 'blue', 'tomato', 'sienna', 'darkorange', 'darkgoldenrod', 'gold', 'olive', 'yellow', 'lawngreen', 
                'palegreen', 'lime', 'cyan', 'dodgerblue', 'indigo', 'magenta', 'deeppink', 'crimson', 'lightpink']
line_width = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

num_method = len(method_name)

N = 100 
mAP_list = np.ones((num_method, N))
mmAP = np.ones((num_method))

for i in range(num_method):
    result = '%s.mat'%method_name[i]
    result = scipy.io.loadmat(result)
    sim_mat = result['sim_mat']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]
    
    print(method_name[i])
    print(sim_mat.shape)
    
    mAP_list[i], mmAP[i] = m.evaluate(sim_mat,query_label,query_cam,gallery_label,gallery_cam,N)
    
    print(mAP_list[i])
    print(mmAP[i])
# Save to  mat for check
result = {'mAP_list': mAP_list, 'mmAP':mmAP}
scipy.io.savemat('CmAP_result.mat', result)

t.draw(mAP_list, mmAP, method_name, line_type, line_color, line_width, N)




