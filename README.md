# T-MAC: A New Evaluation Metric for Person Re-identification

## Prerequisites
- Python
- Scipy
- Numpy

## Getting started
Check the Prerequisites. Then download T_MAC.py and test.py. Next put them in the same directory as you want.

## Part 1: Prepare data
Extract distance, gallery_label, gallery_cam, query_label, and query_cam for the different methods of ReID.

Save the data format as follows:
```
mat:{'distance':distmat,'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_label':query_label,'query_cam':query_cam} # Save to Mat for testing
```

## Part 2: Testing
In test.py, you need to do the following simple operation:
```
method_name = ['ResNet-50', 'DenseNet-121', 'PCB', 'ResNet50_fc512', 'MLFN', 'HA-CNN', 'MobileNetV2_x1_0', 'MobileNetV2_x1_4', 
                'OSNet_x1_0', 'OSNet_x0_75', 'OSNet_x0_5', 'OSNet_x0_25', 'AlignedReID', 'BDB', 'BagTricks', 'AGW', 'ResNet50 + S-ReID',
                'DG-Net',  'FastReID(ResNet50)', 'MGN']
line_type = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.' ,'-', '--', '-.', '-', '--', '-.', '-', '--', '-.', '-', '--']
line_color = ['red', 'green', 'blue', 'tomato', 'sienna', 'darkorange', 'darkgoldenrod', 'gold', 'olive', 'yellow', 'lawngreen', 
                'palegreen', 'lime', 'cyan', 'dodgerblue', 'indigo', 'magenta', 'deeppink', 'crimson', 'lightpink']
line_width = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
```
According to the format and name of ReID method in part 1. You need to modify the parameters of method_name, line_type and line_width.

Next, you can run the shell:
```
python test.py
```

Finally, you can get the curve of T-MAC.
