import json
import os
import numpy as np

from PIL import Image
from tqdm import tqdm
from glob import glob

from .dataset import Dataset
from .video import Video

attr_map = {
    # ----------------------------------------------------------------------
    # A. 车辆 (Car) 序列
    # ----------------------------------------------------------------------
    'car1': ['BC', 'IV', 'CM', 'SV', 'ARC'],
    'car1_s': ['BC', 'IV', 'CM', 'SV', 'ARC'],
    'car2': ['BC', 'IV', 'CM', 'SV', 'ARC'],
    'car2_s': ['BC', 'IV', 'CM', 'SV', 'ARC'],
    'car3': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car3_s': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car4': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car4_s': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car5': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car6': ['FM', 'POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car7': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car8': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car9': ['FM', 'POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car10': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car11': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car12': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car13': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car14': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car15': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car16': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car17': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'car18': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],

    # ----------------------------------------------------------------------
    # B. 人物 (Person) 序列
    # ----------------------------------------------------------------------
    'person1': ['LR', 'IV', 'SV', 'ARC'],
    'person1_s': ['LR', 'IV', 'SV', 'ARC'],
    'person2': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person2_s': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person3': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person3_s': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person4': ['IV', 'VC', 'SV', 'ARC'],
    'person5': ['IV', 'VC', 'SV', 'ARC'],
    'person6': ['IV', 'VC', 'SV', 'ARC'],
    'person7': ['IV', 'VC', 'SV', 'ARC'],
    'person8': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person9': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person10': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person11': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person12': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person13': ['LR', 'IV', 'VC', 'SV', 'ARC'],
    'person14': ['POC', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'person15': ['POC', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'person16': ['POC', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'person17': ['POC', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'person18': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'person19': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'person20': ['POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'person21': ['LR', 'IV', 'SV', 'ARC'],
    'person22': ['LR', 'IV', 'SV', 'ARC'],
    'person23': ['LR', 'IV', 'SV', 'ARC'],

    # ----------------------------------------------------------------------
    # C. 无人机 (UAV) 序列 (UAV123@10fps 子集核心)
    # ----------------------------------------------------------------------
    'uav1': ['IV', 'SV', 'ARC'],
    'uav2': ['LR', 'IV', 'SV', 'ARC'],
    'uav3': ['IV', 'SV', 'ARC', 'VC'],  # 假设 uav3 为主序列
    # 如果 loaddata() 拆分 uav3
    # 'uav3_1': ['SV', 'ARC', 'VC', 'IV'],
    # 'uav3_2': ['SV', 'ARC', 'VC', 'IV'],
    'uav4': ['FM', 'BC', 'IV', 'VC', 'SV', 'ARC'],
    'uav5': ['LR', 'IV', 'CM', 'SV', 'ARC'],
    'uav6': ['LR', 'IV', 'CM', 'SV', 'ARC'],
    'uav7': ['LR', 'FM', 'IV', 'CM', 'SV', 'ARC'],
    'uav8': ['LR', 'FM', 'IV', 'CM', 'SV', 'ARC'],

    # ----------------------------------------------------------------------
    # D. 其它序列 (Truck, Bike, Boat, Building, Group, Wakeboard)
    # ----------------------------------------------------------------------
    'truck1': ['FOC', 'SOB', 'IV', 'SV', 'ARC'],
    'truck2': ['FOC', 'SOB', 'IV', 'SV', 'ARC'],
    'truck3': ['FOC', 'SOB', 'IV', 'SV', 'ARC'],
    'truck4': ['FOC', 'SOB', 'IV', 'SV', 'ARC'],

    'bike1': ['LR', 'IV', 'SV', 'ARC'],
    'bike2': ['LR', 'IV', 'SV', 'ARC'],
    'bike3': ['LR', 'IV', 'SV', 'ARC'],

    'bird1': ['IV', 'VC', 'SV', 'ARC'],

    'boat1': ['LR', 'IV', 'VC', 'CM', 'SV', 'ARC'],
    'boat2': ['LR', 'IV', 'VC', 'CM', 'SV', 'ARC'],
    'boat3': ['LR', 'IV', 'VC', 'CM', 'SV', 'ARC'],
    'boat4': ['LR', 'IV', 'VC', 'CM', 'SV', 'ARC'],
    'boat5': ['LR', 'IV', 'VC', 'CM', 'SV', 'ARC'],
    'boat6': ['LR', 'IV', 'VC', 'CM', 'SV', 'ARC'],
    'boat7': ['LR', 'IV', 'VC', 'CM', 'SV', 'ARC'],
    'boat8': ['LR', 'IV', 'VC', 'CM', 'SV', 'ARC'],
    'boat9': ['LR', 'IV', 'VC', 'CM', 'SV', 'ARC'],

    'group1': ['POC', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'group2': ['POC', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],
    'group3': ['POC', 'BC', 'IV', 'VC', 'CM', 'SOB', 'SV', 'ARC'],

    'building1': ['FOC', 'IV', 'SV', 'ARC'],
    'building2': ['FOC', 'IV', 'SV', 'ARC'],
    'building3': ['FOC', 'IV', 'SV', 'ARC'],
    'building4': ['FOC', 'IV', 'SV', 'ARC'],
    'building5': ['FOC', 'IV', 'SV', 'ARC'],

    'wakeboard1': ['POC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
    'wakeboard2': ['POC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
    'wakeboard3': ['POC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
    'wakeboard4': ['POC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
    'wakeboard5': ['POC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
    'wakeboard6': ['POC', 'BC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
    'wakeboard7': ['POC', 'BC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
    'wakeboard8': ['POC', 'BC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
    'wakeboard9': ['POC', 'BC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
    'wakeboard10': ['POC', 'BC', 'SOB', 'IV', 'VC', 'SV', 'ARC'],
}

def loaddata():
    path=' '

    name_list=os.listdir(path+'/data_seq/')
    name_list.sort()
    a = len(name_list)  # 改为真实长度
    b=[]
    for i in range(a):
        b.append(name_list[i])
    c=[]
    
    for jj in range(a):
        imgs=path+'/data_seq/'+str(name_list[jj])
        txt=path+'/anno/'+str(name_list[jj])+'.txt'
        video_name = name_list[jj]
        current_attr = attr_map.get(video_name, [])
        bbox=[]
        f = open(txt)               # 返回一个文件对象
        file= f.readlines()
        li=os.listdir(imgs)
        li.sort()
        for ii in range(len(file)):
            li[ii]=name_list[jj]+'/'+li[ii]
    
            line = file[ii].strip('\n').split(',')
            
            try:
                line[0]=int(line[0])
            except:
                line[0]=float(line[0])
            try:
                line[1]=int(line[1])
            except:
                line[1]=float(line[1])
            try:
                line[2]=int(line[2])
            except:
                line[2]=float(line[2])
            try:
                line[3]=int(line[3])
            except:
                line[3]=float(line[3])
            bbox.append(line)
            
        if len(bbox)!=len(li):
            print (jj)
        f.close()
        c.append({'attr':current_attr,'gt_rect':bbox,'img_names':li,'init_rect':bbox[0],'video_dir':name_list[jj]})
        
    d=dict(zip(b,c))

    return d

class UAVVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(UAVVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)


class UAV10Dataset(Dataset):
    """
    Args:
        name: dataset name, should be 'UAV123', 'UAV20L'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(UAV10Dataset, self).__init__(name, dataset_root)
        meta_data = loaddata()

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = UAVVideo(video,
                                          dataset_root+'/data_seq',
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'])

        # set attr
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)

