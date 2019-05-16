import numpy as np 
import cv2

def points2heatmap(points, shape = (1024, 1920), r = 20):
    map = np.zeros(shape, dtype = np.float32)
    for point in points:
        if point[0] < shape[0] and point[1] < shape[1]:
            map[point[0], point[1]] = 2 * r
    map = cv2.GaussianBlur(map, (2 * r - 1, 2 * r - 1), 0)
    tmp = map.max()
    if tmp > 0:
        map /= tmp
    return map

def heatmap2points(map, maxsize = 1, r = 40, need_probability = False):
    result = []
    probability = []
    #map = cv2.GaussianBlur(map, (2 * r - 1, 2 * r - 1), 0)
    r -= 1
    h, w = map.shape
    threshold = 0.2
    for _ in range(maxsize):
        map_max = map.max()
        if map_max < threshold:
            break
        index = np.unravel_index(np.argmax(map, axis=None), map.shape)
        result.append(index)
        probability.append(map_max)
        map[max(0, index[0] - r) : min(h, index[0] + r), max(0, index[1] - r) : min(w, index[1] + r)] = 0
    
    if need_probability:
        return result, probability
    else:
        return result
