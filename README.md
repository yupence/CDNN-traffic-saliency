# CDNN:traffic driving saliency & eye tracking dataset 
Paper:How Do Drivers Allocate Their Potential Attention? Driving Fixation Prediction via Convolutional Neural Networks, 2019

Authors:Tao Deng, Hongmei Yan, Long Qin, Thuyen Ngo, B. S. Manjunath

# How Do Drivers Allocate Their Potential Attention?



## Eye tracking dataset
The eye tracking dataset is released [link](https://pan.baidu.com/s/1zyxvEQiMkmOkxmyDlDv0xA);     Password: i5q7

The folder includes: fixdata, traffic videos, traffic frames.You need to extract each frame from videos and put it in traffic frames folder.

use extract_frames.py to get all frames.

use gen_fixation.py to generate some fixation maps.

use evaluation.py to evaluate the model's performance


## CDNN requirements
* Pytorch 1.4.0

* Python 3.7

Contact the Author: tdeng@swjtu.edu.cn
