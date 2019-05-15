#!/usr/bin/env python

'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import video, fit, track_backside
import sys
sys.path.append("..")
import annotate

def hisEqulColor(img0):
    return img0.copy()
    ycrcb = cv.cvtColor(img0, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    cv.equalizeHist(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    img = cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR)
    return img

track = []

def show_track():
    global track, frame, circle_list
    #print(circle_list)
    if len(track) > 5:
        for i in range(len(track)):
            
            '''
            [x, y, z] = annotate.calc_coordinate(track[i][1 : 3], circle_list[frame - len(track) + i][:2])
            track[i] = [track[i][0], x, y, z]
            '''
            
            track[i] = [track[i][0], track[i][1] * 0.0013, track[i][2] * 0.0013, 0]
        #print(track)
        fit.fit_track(track)

def get_track(img_gray, img, flow, vis):
    global frame_count, flow_sum, track

    fx, fy = flow[:,:,0], flow[:,:,1]
    v = np.sqrt(fx*fx+fy*fy)
    v_gain = v / flow_sum * frame_count
    v_max = v_gain.max() 

    flow_sum += np.maximum(v, 1)
    frame_count += 1
    
    print(v_max)
    if v_max < 48:
        if len(track) > 0:
            show_track()
            track = []
        return vis
    

    (y_max, x_max) = np.unravel_index(np.argmax(v_gain), v.shape)
    size = 20
    #cv.rectangle(vis, (x_max - size, y_max - size), (x_max + size, y_max + size), (255, 0, 0), 1)
    #print(x_max, y_max)
    
    tmp = cv.cvtColor(np.array((img_gray > 50) * (v > 1), dtype = np.uint8) * 255, cv.COLOR_GRAY2BGR)
    #vis = tmp
    h, w = tmp.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype = np.uint8)
    cv.floodFill(tmp, mask, (x_max, y_max), (0, 0, 255))
    mask = mask[1 : h + 1, 1 : w + 1]
    _, x = np.where(mask == 1)
    x_max = x.max()
    mask = mask[:, x_max]
    y = np.where(mask == 1)
    y_max = y[0].max()
    y_shape, x_shape = vis.shape[:2]
    print(x_max, y_max)
    if len(track) > 1:
        #
        if (x_max - track[-1][1]) * (track[-1][1] - track[-2][1]) < 0 \
        or (y_max - track[-1][2]) * (track[-1][2] - track[-2][2]) < 0 \
        or 2 * track[-1][1] - track[-2][1] < 0 \
        or 2 * track[-1][1] - track[-2][1] >= x_shape \
        or 2 * track[-1][2] - track[-2][2] < 0 \
        or 2 * track[-1][2] - track[-2][2] >= y_shape:
            show_track()
            track = []
            return vis

    print(x_max, y_max)
    cv.circle(vis, (x_max, y_max), 2, (0, 0, 255), 1, 0)
    track.append([time, float(x_max), float(y_max)])
    return vis


def draw_flow(img_gray, img, flow, step=10000):
    global mask, time, delta_t
    
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    #vis = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    #vis = cv.cvtColor(img_gray * mask, cv.COLOR_GRAY2BGR)
    vis = img
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    vis = get_track(img_gray, img, flow, vis)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    (y_max, x_max) = np.unravel_index(np.argmax(v), v.shape)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

def denoise(img):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5, 5))
    img = cv.erode(img, kernel)
    return cv.dilate(img, kernel)
    #return cv.fastNlMeansDenoising(img)

def ColourDistance(img, color = (255, 255, 255)):
     img_B, img_G, img_R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
     B, G, R = color
     rmean = (img_R + R) / 2.
     dR = img_R - R
     dG = img_G - G
     dB = img_B - B
     return np.sqrt((2+rmean / 256.) * (dR**2) + 4  *(dG**2) + (2 + (255 - rmean) / 256.) * (dB**2))


if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
        backside_path = sys.argv[2]
        #annotate.check()
        #circle_list = track_backside.track_backside(backside_path)
    except IndexError:
        fn = 0
        circle_list = None

    #optical_flow = cv.DualTVL1OpticalFlow_create()

    cam = video.create_capture(fn)
    ret, prev = cam.read()
    prev = hisEqulColor(prev)

    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    #prevgray = denoise(prevgray)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    frame = 0
    frame_count = 1
    flow_sum = np.ones((prev.shape[0], prev.shape[1]), dtype=float)
    time = 0
    delta_t = 1 / 500.
    while True:
        ret, img = cam.read()
        img_hist = hisEqulColor(img)
        
        #print(img.shape)
        #img = cv.resize(img, (1280, 720))

        #if frame < 225:
        
        
        #if frame < 3561:
        '''
        if frame < 2650:
            prevgray = cv.cvtColor(img_hist, cv.COLOR_BGR2GRAY)
            continue 
        '''
        
        print('%d :'%(frame), end = '')
        time += delta_t

        gray = cv.cvtColor(img_hist, cv.COLOR_BGR2GRAY)
        #gray = cv.equalizeHist(gray)
        #gray = denoise(gray)
        
        #flow = optical_flow.calc(prevgray, gray, None)
        
        
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = 0.5,\
         levels = 5, winsize = 20, iterations = 5, poly_n = 5, poly_sigma = 1.2, flags = 0)


        threshold = 50
        threshold_dis = 30
        #mask = ColourDistance(img) < threshold_dis
        #mask = img.min(axis = 2) > threshold
        mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY) > threshold
        flow *= np.expand_dims(mask, axis=2)
        flow = cv.GaussianBlur(flow, (7, 7), 0)
        
        prevgray = gray 

        result = draw_flow(gray, img, flow)
        
        cv.imshow('flow', result)
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)

        ch = cv.waitKey(0)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])
        
        
        
        frame += 1

    cv.destroyAllWindows()
