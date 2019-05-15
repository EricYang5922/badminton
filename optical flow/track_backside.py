import numpy as np
import cv2 as cv
import video


def hisEqulColor(img0):
    return img0.copy()
    ycrcb = cv.cvtColor(img0, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    cv.equalizeHist(channels[0], channels[0])
    cv.merge(channels, ycrcb)
    img = cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR)
    return img

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def track_backside(path):
    cam = video.create_capture(path)
    ret, prev = cam.read()
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

    frame_list, circles_list = [], []

    while True:
        ret, img = cam.read()
        if not ret:
            break

        '''
        img = np.array(img, dtype=np.float)
        count = 1

        for i in range(10):
            count += 1
            ret, img0 = cam.read()
            img += img0

        img = np.array(img / count, dtype=np.uint8)
        '''
        cimg = img.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.medianBlur(gray, 5)

        
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = 0.5,\
         levels = 5, winsize = 20, iterations = 5, poly_n = 5, poly_sigma = 1.2, flags = 0)
        
        threshold = 50
        threshold_dis = 30
        #mask = ColourDistance(img) < threshold_dis
        #mask = img.min(axis = 2) > threshold
        mask = gray > threshold
        flow *= np.expand_dims(mask, axis=2)

        flow = cv.GaussianBlur(flow, (7, 7), 0)
        img = img * (flow.max(2) > 1)

        prevgray = gray

        '''
        threshold = 50
        img[img <= threshold] = 0
        img[img > threshold] = 255

        cv.imshow('image', img)
        cv.waitKey(0)
        '''
        

        #cimg = img.copy()

        circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1.7, 50, np.array([]), 100, 30, 14, 30)

        frame_list.append(cimg)
        circles_list.append(circles[0])


    circle_list = list(range(len(frame_list)))
    t = 0

    for i in range(len(frame_list) - 1, -1, -1):
        circles = circles_list[i]
        if circles is not None and len(circles)  == 1: 
            circle_list[i] = circles[0]
            t = i
            break

    for i in range(t + 1, len(frame_list)):
        circles = circles_list[i]
        if circles is None:
            break
        dist = [distance(circle_list[i - 1][0], circle_list[i - 1][1], circle[0], circle[1]) for circle in circles]
        dist = np.array(dist)
        index = dist.argmin()
        circle_list[i] = circles[index]

    for i in range(t - 1, -1, -1):
        circles = circles_list[i]
        if circles is None:
            break
        dist = [distance(circle_list[i + 1][0], circle_list[i + 1][0], circle[0], circle[1]) for circle in circles]
        dist = np.array(dist)
        index = dist.argmin()
        circle_list[i] = circles[index]

    for i in range(len(frame_list)):
        circle = circle_list[i]
        cimg = frame_list[i]
        
        if circle is not None and len(circle)  == 3: # Check if circles have been found and only then iterate over these and add them to the image 
            x, y, r = circle
            cv.circle(cimg, (x, y), r, (0, 0, 255), 1, cv.LINE_AA)
            cv.circle(cimg, (x, y), 1, (0, 255, 0), 1, cv.LINE_AA)  # draw center of circle

            
            cv.imshow("detected circles", cimg)
            key = cv.waitKey(0)
            if key & 0xFF == 27:
                break
            
            
    #print(circle_list)
    return circle_list

if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0