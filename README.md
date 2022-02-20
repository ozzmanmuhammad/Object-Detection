# Vision - Object Detection & Real-Time Prediction in Browser.
Deep learning experiments for the Object Detection, with bounding boxes and COCO human pose keypoints.

## Overview
In this project, I built an object detection model using <a href="https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1" target="_blank"><b>'CenterNet HourGlass104 Keypoints 512x512'</b></a> which was pre-trained on COCO-2017 dataset containing 80 different object categories. This model also contained human pose keypoints which were also used during visualization of predictions.

here are many other pretrained models present on <a href="https://tfhub.dev/tensorflow/collections/object_detection/1" target="_blank">tensorflow-hub object detection</a> and 
I used CenterNet because of it speed and accuracy rate, while in-case of live predictions in browser in used MobileNet / <a href="https://learn.ml5js.org/#/reference/object-detector" target="_blank">COCO-SSD</a> which is light, less complex, faster inference speed but less accurate then CenterNet. (<a href="https://paperswithcode.com/paper/centernet-object-detection-with-keypoint" target="_blank">CenterNet Paper</a>)

## Code and Resources Used
- Python Version: 3.8
- Tensorflow Version: 2.7.0
- Tensorflow. JS
- Packages: numpy, sklearn, matplotlib, seaborn
- Editor:  Google Colab

## Dataset and EDA
The CenterNet model was pretrained on COCO-2017 datasets, but there are latest models present in the tensorflow-hub 
 which are trained on COCO-2020 datasets. The <a href="https://cocodataset.org/#home">COCO</a> have many different datasets ranging from
 keypoints to object segmentation. The CenterNet model was pretrained on following categories:
                                                        

|               | Experiment #1 | 
| ------------- |:-------------------:|
| Dataset source|  COCO| 
| Classes |  80| 
|Data Augmentation|	No|
|Data loading|	No|

some examples from the Train datasets:

 <em>
1: person
2: bicycle
3: car
4: motorcycle
5: airplane
6: bus
7: train
8: truck
9: boat
10: traffic light
11: fire hydrant
12: stop sign
13: parking meter
14: bench
15: bird
16: cat
17: dog
18: horse
19: sheep
20: cow
21: elephant
22: bear
23: zebra
24: giraffe
25: backpack
26: umbrella
27: handbag
28: tie
29: suitcase
30: frisbee
31: skis
32: snowboard
33: sports ball
34: kite
35: baseball bat
36: baseball glove
37: skateboard
38: surfboard
39: tennis racket
40: bottle
41: wine glass
42: cup
43: fork
44: knife
45: spoon
46: bowl
47: banana
48: apple
49: sandwich
50: orange
51: broccoli
52: carrot
53: hot dog
54: pizza
55: donut
56: cake
57: chair
58: couch
59: potted plant
60: bed
61: dining table
62: toilet
63: tv
64: laptop
65: mouse
66: remote
67: keyboard
68: cell phone
69: microwave
70: oven
71: toaster
72: sink
73: refrigerator
74: book
75: clock
76: vase
77: scissors
78: teddy bear
79: hair drier
80: toothbrush.
</em>

## Model Building and Performance

The experiments was done on CNN architecture model, which was trainied 10 epochs (less epochs to avoid overfitting as it was trained only on 2,388 images)

|               | Experiment #1 | 
| ------------- |:-------------------:|
|Architecture|	CNN|
|Big-Model|	CenterNet|
|Lite-Model|	MobileNet/ COCONet|
|Model Returns|	|
|1|	detection_classes|
|2|	detection_keypoint_scores|
|3|	detection_boxes|
|4|	num_detections|
|5|	detection_keypoints|
|6|	detection_scores|


<br/><br/>

## Model Predictions
After loading and preprocessing the images, I used <a href="https://github.com/tensorflow/models">Tensorflow object detection API</a> for predictions and visulization of boxes and keypoints otherwise I would've to create custom functions to create bounding boxes and pose keypoints which would've taken alot of time.
<br/> Some of the predictions examples are: <br/><br/>
<img src="https://github.com/ozzmanmuhammad/ozzmanmuhammad.github.io/blob/main/assets/images/project/OD_pred_1.png"  alt="">
<img src="https://github.com/ozzmanmuhammad/ozzmanmuhammad.github.io/blob/main/assets/images/project/pred3.png"  alt="">
<img src="https://github.com/ozzmanmuhammad/ozzmanmuhammad.github.io/blob/main/assets/images/project/pred4.png"  alt="">
<img src="https://github.com/ozzmanmuhammad/ozzmanmuhammad.github.io/blob/main/assets/images/project/pred5.png"  alt="">
<img src="https://github.com/ozzmanmuhammad/ozzmanmuhammad.github.io/blob/main/assets/images/project/pred6.png"  alt="">

Future experiments can be done to make object segmentation and
to detect and read specfic objects like car license plates.

## Real Time Predictions in Browser.
It uses the COCO-SSD model pre-trained on 80-Categories which was developed by the TensorFlow.js team in 2018, where it is currently maintained on
TensorFlow.js. In this project I'm using ml5 library to load and detect objects from images or webcam frames as input and p5.JS to create boxes and labels.
To try it yourself please visit project page on my 
<a href="https://ozzmanmuhammad.github.io/project-Object_detection.htmll" target="_blank">"Portfolio."</a>
