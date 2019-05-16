import numpy as np
import cv2
import os
import torch
from . import translate, heatmap, visualize, calc_3D

class video():
    def __init__(self, path, start_frame = 0, max_frame = -1, enhancement = 1):
        self.enhancement = enhancement
        if os.path.isdir(path):
            self.mode = 'images'
            self.images = []
            for filename in os.listdir(path):
                if filename.endswith('jpg') or filename.endswith('png'):
                    self.images.append(os.path.join(path, filename))   
            self.images = sorted(self.images)[start_frame :]
            if max_frame > 0 and max_frame < len(self.images):
                self.images = self.images[:max_frame]
        elif os.path.isfile(path):
            self.mode = 'video'
            self.max_frame = max_frame
            self.cap = cv2.VideoCapture(path)
            for i in range(start_frame):
                success, image = self.cap.read()



    def next_frame(self):
        if self.mode == 'images':
            if len(self.images) == 0:
                return None
            else:
                image = cv2.imread(self.images[0])
                h, w = image.shape[:2]
                new_h, new_w = h // 128 * 128, w // 128 * 128
                image = image[: new_h, : new_w]
                if self.enhancement:
                    image = np.clip(1.4 * image, 0, 255).astype(np.uint8)
                self.images.pop(0)
                return image
        elif self.mode == 'video':
            if self.max_frame == 0:
                self.cap.release()
                return None
            success, image = self.cap.read()
            if success:
                h, w = image.shape[:2]
                new_h, new_w = h // 128 * 128, w // 128 * 128
                image = image[: new_h, : new_w]
                image = np.clip(self.enhancement * image, 0, 255).astype(np.uint8)
                self.max_frame -= 1
                return image
            else: 
                self.cap.release()
                return None
        else:
            return None


class video_detector():
    def __init__(self, video_data, neighbors, net, device):
        self.video_data = video_data
        self.neighbors = neighbors
        self.net = net
        self.device = device

        image_list = []
        for i in range(neighbors):
            image_list.append(video_data.next_frame())
        self.image_set = np.concatenate(image_list, axis = 2)

        h, w = self.image_set.shape[:2]
        self.heatmap_set = np.zeros((h, w, neighbors), dtype = np.float32)

        self.image_set = translate.img2torch(self.image_set)
        self.heatmap_set = translate.np2torch(self.heatmap_set)
        self.frame_num = 0

    def detect(self, save_path):
        current_frame = self.video_data.next_frame()
        if current_frame is None:
            return None
        
        torch_image = translate.img2torch(current_frame)
        self.image_set = torch.cat((self.image_set, torch_image), 0)
        data = torch.unsqueeze(torch.cat((self.image_set, self.heatmap_set), 0), 0)

        with torch.no_grad():
            data = data.to(self.device)
            result = self.net(data)[0].cpu().data[0]
            result_heatmap = result.numpy()

        points, probability = heatmap.heatmap2points(result_heatmap[0].copy(), 3, need_probability=True)
        

        #print(points)

        img = visualize.label_points(current_frame, points)
        result_heatmap = translate.np2heatmap(result_heatmap[0])
        result_heatmap = cv2.addWeighted(img, 0.6, result_heatmap, 0.4, 0)
        img = np.concatenate((img, result_heatmap), 0)
        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, 'video_%08d.jpg'%(self.frame_num)), img)
        
        #print('\r', 'handling frame %d'%(self.frame_num), end = '')
        self.frame_num += 1

        self.image_set = self.image_set[3:]
        self.heatmap_set = torch.cat((self.heatmap_set[1:], result), 0)
        return points, probability, img



def inference_video(video_path, neighbors, net, device, save_path):
    net.eval()
    video_data = video(video_path)
    detector = video_detector(video_data, neighbors, net, device)
    while True:
        ret = detector.detect(save_path)
        if ret is None:
            break

def calc_distance(coord1, coord2):
    return np.sqrt(np.square(np.array(coord1) - np.array(coord2)).sum()) + 1e-5

def video_3D(video_datas, mats, neighbors, net, device, save_path):
    net.eval()
    video_data0, video_data1 = video_datas
    detector0 = video_detector(video_data0, neighbors, net, device)
    detector1 = video_detector(video_data1, neighbors, net, device)

    #save_path0, save_path1 = None, None

    num = 0
    out = None

    coordinate_list = []
    candidate_list = []

    while True:
        ret0 = detector0.detect(None)
        ret1 = detector1.detect(None)
        if ret0 is None or ret1 is None:
            break
        print('handling frame %d coordinate:'%(num), end = '')

        if len(ret0[0]) > 0 and len(ret1[0]) > 0:
            coordinate_candidate = []
            for i in range(len(ret0[0])):
                for j in range(len(ret1[0])):
                    point0, point1 = ret0[0][i], ret1[0][j]
                    coordinate = calc_3D.calc_3D(mats, (point0, point1))
                    if len(coordinate_list) > 0:
                        dis = calc_distance(coordinate_list[-1][1:], coordinate)
                    else:
                        dis = 1.
                    if abs(coordinate[0, 0]) < (670 + 20) and abs(coordinate[0, 1]) < (305 + 20) and coordinate[0, 2] > (0-20):
                        coordinate_candidate.append([coordinate, ret0[1][i] * ret1[1][j] / dis])
            if len(coordinate_candidate) > 0:
                coordinate_candidate.sort(key = lambda x: x[1], reverse = True)
                #print(coordinate_candidate)
                coordinate = coordinate_candidate[0][0]
                #print('points: ', point0, point1, end = '')
                print(coordinate)
                coordinate_list.append([num, coordinate[0, 0], coordinate[0, 1], coordinate[0, 2]])
                for coordinate in coordinate_candidate:
                    candidate_list.append([num, coordinate[0][0, 0], coordinate[0][0, 1], coordinate[0][0, 2], coordinate[1]])
            else:
                print('not found')
        else:
            print('not found')
        
        img0, img1 = ret0[-1], ret1[-1]
        img = np.concatenate((img0, img1), 1)   
        #cv2.imwrite(os.path.join(save_path, 'video_%08d.jpg'%(num)), img)
        num += 1

    if out is not None:
        out.release()

    return coordinate_list, candidate_list




        



            
