#coding:utf8
import os
import cv2
import random
from PIL import  Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from utils import translate, load_json, heatmap
import torch

class shuttle(data.Dataset):
    
    def __init__(self, path, neighbor, mode = 'detection', train=True):
        self.train = train
        self.neighbor = neighbor
        self.data = []
        self.mode = mode

        dir_list = []

        for file_name in os.listdir(path):
            if os.path.isdir(os.path.join(path, file_name)):
                dir_list.append(os.path.join(path, file_name))
        
        dir_list = sorted(dir_list)
        #np.random.shuffle(dir_list)    
        size = len(dir_list)

        if train:
            self.segments = dir_list[: int(size * 0.8)]
        else:
            self.segments = dir_list[int(size * 0.8) :]   
            '''
            self.segments = dir_list[-3 :]
            self.segments = dir_list[8 :]
            '''

        print(self.segments)
        #self.segments = dir_list

        for segment_path in self.segments:
            data = []
            for filename in os.listdir(segment_path):
                if filename.endswith('jpg') or filename.endswith('png'):
                    data.append(os.path.join(segment_path, filename))
            data = sorted(data)
            if mode == 'detection':
                for i in range(neighbor, len(data) - neighbor):
                    self.data.append(data[i - neighbor : i + neighbor + 1])
            elif mode == 'tracking':
                for i in range(neighbor, len(data)):
                    tmp = data[i - neighbor : i + 1]
                    for file_name in data[i - neighbor : i]:
                        tmp.append(file_name.replace('.jpg', '.json'))
                    self.data.append(tmp)
        
        
    def __getitem__(self, index):
        '''
        一次返回一段片段
        '''
        if self.mode == 'detection':
            segment_path = self.data[index]
        elif self.mode == 'tracking':
            segment_path = self.data[index][: self.neighbor + 1]
            heatmap_paths = self.data[index][self.neighbor + 1 :]
        img_list = []
        for img_path in segment_path:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            new_h, new_w = h // 128 * 128, w // 128 * 128
            img = img[:new_h, :new_w] 
            img_list.append(img)

        if self.mode == 'tracking':
            for heatmap_path in heatmap_paths:
                points = load_json.load_json(heatmap_path)
                map = heatmap.points2heatmap(points)
                img_list.append(np.expand_dims(map, 2))

        segment = np.concatenate(img_list, axis = 2)  
        data = translate.img2torch(segment, self.train)

        gt_path = segment_path[self.neighbor].replace('.jpg', '.json')     
        points = load_json.load_json(gt_path)
        gt_map = heatmap.points2heatmap(points)
        gt_map = translate.np2torch(gt_map)

        return data, gt_map
    
    def __len__(self):
        return len(self.data)
