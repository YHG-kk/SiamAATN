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

import time #

from glob import glob

from pysot.core.config_aatn import cfg
from pysot.models.model_builder_aatn import ModelBuilderAATN
from pysot.tracker.siamaatn_tracker import SiamAATNTracker
from pysot.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamAATN demo')
parser.add_argument('--config', type=str, default='../experiments/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='../pretrained_models/general_model.pth', help='model name')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')


args = parser.parse_args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, '*.jp*')))
        print("找到图片数量：", len(images))
        if len(images) == 0:
            print("图片没找到！路径是否正确？")
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():

    fps, tic = 0.0, time.time()
    alpha = 0.9

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')


    model = ModelBuilderAATN()


    model = load_pretrain(model, args.snapshot).eval().to(device)


    tracker = SiamAATNTracker(model)
    f = None
    video_is_sequence = False

    first_frame = True
    if args.video_name:
        video_name_full_path = args.video_name.replace('\\', '/')
        video_name = video_name_full_path.split('/')[-1].split('.')[0]


        if not args.video_name.endswith('avi') and \
                not args.video_name.endswith('mp4'):
            video_is_sequence = True
    else:
        video_name = 'webcam'


    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    # 仅当确定是图片序列时，才打开文件
    if video_is_sequence:
        save_dir = './result_bbox'
        os.makedirs(save_dir, exist_ok=True)
        result_path = os.path.join(save_dir, f'{video_name}.txt')

        try:
            f = open(result_path, 'w')
            print(f"✅ 结果将保存到: {result_path}")
        except Exception as e:
            print(f"❌ 无法打开结果文件: {e}")
            sys.exit(1)

    first_frame = True
    try:
        for frame in get_frames(args.video_name):

            toc = time.time()
            inst_fps = 1.0 / (toc - tic)
            fps = alpha * fps + (1 - alpha) * inst_fps
            tic = toc

            if first_frame:
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)

                    if f:
                        f.write(','.join([str(s) for s in init_rect]) + '\n')
                except:
                    exit()
                tracker.init(frame, init_rect)
                first_frame = False
            else:
                outputs = tracker.track(frame)
                bbox = list(map(int, outputs['bbox']))
                bbox_float = outputs['bbox']

                if f:
                    bbox_str = ','.join([f"{s:.2f}" for s in bbox_float])
                    f.write(bbox_str + '\n')

                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)


                cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                cv2.imshow(video_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:

        if f:
            f.close()
            print(f"✅ 文件 {result_path.split('/')[-1]} 已关闭并保存。")



    return

if __name__ == '__main__':
    main()
