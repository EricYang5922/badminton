import numpy as np
import torch
import cv2

def torch2np(data):
    data = data.numpy()
    if data.ndim == 3:
        data = np.transpose(data, (1, 2, 0))
    return data

def np2torch(img):
    if img.ndim == 2:
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    return img

def img2torch(img, augmentation = False):
    img = img.astype(np.float32) / 255.
    if augmentation:
        img += np.random.normal(0, 0.03, img.shape)
    return np2torch(img)

def torch2img(data):
    img = torch2np(data)
    img = (np.clip(img * 255, 0, 255)).astype(np.uint8)
    return img

def np2heatmap(map):
    #print(map.max())
    img = (np.clip(map * 255, 0, 255)).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_JET)

