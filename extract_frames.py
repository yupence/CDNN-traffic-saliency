import cv2  # 导入opencv模块
import os
import time


def video_split(video_path, save_path):
    vc = cv2.VideoCapture(video_path)
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        cv2.imwrite(save_path + "/" + str('%06d' % c) + '.jpg', frame)
        cv2.waitKey(1)
        c = c + 1
        rval, frame = vc.read()


if __name__ == "__main__":
    DATA_DIR = r"home/download/traffic"
    SAVE_DIR = r"home/project/CDNN/CDNN-traffic-saliency/traffic_frames"
    traffic_dict = {"out1": "01", "out2": "02", "out3": "03", "out4": "04", "out5": "05", "out6": "06", "out7": "07",
                    "out8": "08", "out9": "09", "out10": "10", "out11": "11", "out12": "12", "out13": "13",
                    "out14": "14", "out15": "15", "out16": "16"}
    start_time = time.time()
    for parents, dirs, filenames in os.walk(DATA_DIR):
        path = parents.replace("\\", "/")
        f = parents.split("\\")[1]
        save_path = SAVE_DIR + "/"
        for file in filenames:
            file_name = file.split(".")[0]
            save_path_ = save_path + "/" + traffic_dict[file_name]
            if not os.path.isdir(save_path_):
                os.makedirs(save_path_)
            video_path = path + "/" + file
            video_split(video_path, save_path_)
    end_time = time.time()
