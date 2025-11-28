from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
import time

from pysot.core.config_aatn import cfg
from pysot.models.model_builder_aatn import ModelBuilderAATN
from pysot.tracker.siamaatn_tracker import SiamAATNTracker
from pysot.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamAATN webcam demo')
parser.add_argument('--config', type=str, default='../experiments/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='../pretrained_models/general_model.pth', help='model name')
args = parser.parse_args()


def get_frames_from_camera(camera_id=0):

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)


    for _ in range(5):
        cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def main():

    fps, tic = 0.0, time.time()
    alpha = 0.9  # 平滑系数


    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    model = ModelBuilderAATN()
    model = load_pretrain(model, args.snapshot).eval().to(device)

    tracker = SiamAATNTracker(model)


    window_name = "SiamAATN Webcam Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    first_frame = True
    print("🎥 摄像头启动中，请稍候...")

    try:
        for frame in get_frames_from_camera(0):
            toc = time.time()
            inst_fps = 1.0 / (toc - tic)
            fps = alpha * fps + (1 - alpha) * inst_fps
            tic = toc

            if first_frame:
                print("🔲 请用鼠标框选目标，按 ENTER 或 SPACE 确认")
                init_rect = cv2.selectROI(window_name, frame, False, False)
                if sum(init_rect) == 0:
                    print("⚠️ 未选中目标，退出。")
                    break
                tracker.init(frame, init_rect)
                first_frame = False
                continue


            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))


            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 255, 0), 3)
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    except KeyboardInterrupt:
        print("\n🛑 手动中断。")

    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
