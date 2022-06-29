import numpy as np
import cv2
import draw
import matplotlib.pyplot as plt
from torch import nn

if __name__ == "__main__":
    data = np.load(r"./ckpts/cdnn/test.npz")
    output = data['p']
    target = data['t']
    list1 = [100, 300, 700]
    for item in list1:
        output_mask = cv2.resize(output[item],(1280,720))
        target_mask = cv2.resize(target[item],(1280,720))
        num = str(item+5).zfill(6)
        picturePath = r'F:/baidudownload/baidudownload/out2/'+num+'.jpg'
        OsavePath = r'./mask_photo/O'+num+'.jpg'
        draw.draw_picture(picturePath,output_mask,OsavePath)
        TsavePath = r'./mask_photo/T'+num+'.jpg'
        draw.draw_picture(picturePath,target_mask,TsavePath)


