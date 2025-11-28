import os
import sys
import time
import argparse
import functools
sys.path.append("./")

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import UAV10Dataset,UAV20Dataset,VISDRONED2018Dataset
from toolkit.evaluation import OPEBenchmark
from toolkit.visualization import draw_success_precision
import matplotlib as mpl

TRACKER_TO_BOLD = 'SiamAATN' # <-- 在这里指定您想要加粗的名称


if __name__ == '__main__':
    
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']

    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    # parser.add_argument('--dataset_dir', default='',type=str, help='dataset root directory')
    parser.add_argument('--dataset_dir', default=' ', type=str,
                        help='dataset root directory')

    parser.add_argument('--dataset', default='UAV10',type=str, help='dataset name')
    parser.add_argument('--tracker_result_dir',default='', type=str, help='tracker result root')
    parser.add_argument('--trackers',default='general_model', nargs='+')
    parser.add_argument('--attribute', default='ALL', type=str,
                        help='Attribute to test (e.g., LR, SV, CM, or ALL for full set)')
    parser.add_argument('--vis', default='',dest='vis', action='store_true')
    parser.add_argument('--show_video_level', default=' ',dest='show_video_level', action='store_true')
    parser.add_argument('--num', default=20, type=int, help='number of processes to eval')
    args = parser.parse_args()


    tracker_dir = os.path.join(args.tracker_result_dir, args.dataset)
    trackers = args.trackers



    trackers = [x.split('/')[-1] for x in trackers]

    root = args.dataset_dir

    print(">> Dataset root path:", root)
    print(">> Tracker result dir:", tracker_dir)
    print(">> Trackers:", trackers)
    if not os.path.exists(root):
        print("❌ Dataset path does not exist:", root)
    if not os.path.exists(tracker_dir):
        print("❌ Tracker result dir does not exist:", tracker_dir)

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'UAV123_10fps' in args.dataset:
        dataset = UAV10Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        print(success_ret)
        print(precision_ret)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                            # name=dataset.name,
                            name='UAV123@10fps',
                            videos=videos,
                            attr=attr,
                            precision_ret=precision_ret,
                            bold_name=TRACKER_TO_BOLD)
    elif 'UAV123_20L' in args.dataset:
        dataset = UAV20Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)

        print(success_ret)
        print(precision_ret)


        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                            # name=dataset.name,
                            name='UAV20L',
                            videos=videos,
                            attr=attr,
                            precision_ret=precision_ret,
                            bold_name=TRACKER_TO_BOLD)


    elif 'VISDRONED' in args.dataset:
        dataset = VISDRONED2018Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=18):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=18):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        print(success_ret)
        print(precision_ret)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                            name=dataset.name,
                            videos=videos,
                            attr=attr,
                            precision_ret=precision_ret,
                            bold_name=TRACKER_TO_BOLD)



 