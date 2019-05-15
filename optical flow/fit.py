import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

g = 9.8
 
#直线方程函数
def f_1(x, A, B):
    return A*x + B
 
#二次曲线方程
def f_2(x, A, B, C):
    return A*x*x + B*x + C
 
#三次曲线方程
def f_3(x, A, B, C, D):
    return A*x*x*x + B*x*x + C*x + D

def f_4(x, A, B, C, D, E):
    return A*x*x*x*x + B*x*x*x + C*x*x + D*x + E

def f_x(t, A, t0, C1, k1):
    return np.log(A * (t - t0) + C1) / A + k1

def f_y(t, B, t0, C2, k2):
    return np.log(np.exp(2 * np.sqrt(B * g) * (t - t0) + C2) - 1) / B - np.sqrt(g / B) * (t - t0) + k2
 
def linear_fit(x, y):
    A, B = optimize.curve_fit(f_1, x, y)[0]
    return A, B

def quadratic_fit(x, y):
    A, B, C = optimize.curve_fit(f_2, x, y)[0]
    return A, B, C

def cubic_fit(x, y):
    A, B, C, D= optimize.curve_fit(f_3, x, y)[0]
    return A, B, C, D

def quartic_fit(x, y):
    A, B, C, D, E = optimize.curve_fit(f_4, x, y)[0]
    return A, B, C, D, E

def x_fit(x, y):
    try:
         A, t0, C1, k1 = optimize.curve_fit(f_x, x0, y0, maxfev = 10000)[0]
    except:
        return None
    return A, t0, C1, k1

def y_fit(x, y):
    try:
        t, B, t0, C2, k2 = optimize.curve_fit(f_y, x0, y0, maxfev = 100000)[0]
    except:
        return None
    return t, B, t0, C2, k2

def draw_curve(x, y_list, color_list, title = 'curve fit', xlabel = 'x', ylabel = 'y'):
    for i in range(len(y_list)):
        plt.plot(x, y_list[i], color_list[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def fit_track(track):
    color_list = ['blue', 'green', 'purple', 'yellow']
    track = np.array(track)
    plt.figure()
    plt.subplot(3, 3, 1)
    time, x_coordinates, x_speeds = plot_test(track[:, 0], track[:, 1], f_x, xlabel='time', ylabel='x')
    plt.subplot(3, 3, 2)
    time, y_coordinates, y_speeds = plot_test(track[:, 0], track[:, 2], f_y, xlabel='time', ylabel='y')
    plt.subplot(3, 3, 3)
    time, z_coordinates, z_speeds = plot_test(track[:, 0], track[:, 3], f_x, xlabel='time', ylabel='z')
    plt.subplot(3, 3, 4)
    draw_curve(time, x_speeds, color_list, xlabel='time', ylabel='speed_x')
    plt.subplot(3, 3, 5)
    draw_curve(time, y_speeds, color_list, xlabel='time', ylabel='speed_y')
    plt.subplot(3, 3, 6)
    draw_curve(time, z_speeds, color_list, xlabel='time', ylabel='speed_z')
    
    speeds = ((np.array(x_speeds) ** 2 + np.array(y_speeds) ** 2 + np.array(z_speeds) ** 2) ** 0.5).tolist()
    plt.subplot(3, 3, 7)
    draw_curve(time, speeds, color_list, xlabel='time', ylabel='speed')
    
    plt.show()

def plot_test(x, y, func, title = 'curve fit', xlabel = 'x', ylabel = 'y'):

    global track
 
    #plt.figure()
 
    #拟合点
    #x0 = [1, 2, 3, 4, 5]
    #y0 = [1, 3, 8, 18, 36]
    x_min, x_max = x.min(), x.max()
    delta_x = x_max - x_min
    x_start, x_end, step = x_min - delta_x, x_max + delta_x, delta_x / 100. 
 
    #绘制散点
    plt.scatter(x[:], y[:], 25, "red")

    x0 = np.arange(x_start, x_end, step)
 
    #直线拟合与绘制
    A1, B1 = optimize.curve_fit(f_1, x, y)[0]
    y1 = A1 * x0 + B1
    speed_y1 = x0 + A1 - x0
    plt.plot(x0, y1, "blue")
 
    #二次曲线拟合与绘制
    A2, B2, C2 = optimize.curve_fit(f_2, x, y)[0]
    x0 = np.arange(x_start, x_end, step)
    y2 = A2 * x0 ** 2 + B2 * x0 + C2 
    speed_y2 = 2 * A2 * x0 + B2
    plt.plot(x0, y2, "green")
 
    #三次曲线拟合与绘制
    A3, B3, C3, D3= optimize.curve_fit(f_3, x, y)[0]
    y3 = A3 * x0 ** 3 + B3 * x0 ** 2 + C3 * x0 + D3 
    speed_y3 = 3 * A3 * x0 ** 2 + 2 * B3 * x0 + C3
    print(A3, B3, C3, D3)
    plt.plot(x0, y3, "purple")

    A4, B4, C4, D4, E4 = optimize.curve_fit(f_4, x, y)[0]
    x4 = np.arange(x_start, x_end, step)
    y4 = f_4(x4, A4, B4, C4, D4, E4)
    speed_y4 = 4 * A4 * x0 ** 3 + 3 * B4 * x0 ** 2 + 2 * C4 * x0 + D4
    print(A4, B4, C4, D4, E4)
    plt.plot(x0, y4, "yellow")

    '''
    A, t0, C1, k1 = optimize.curve_fit(f_x, x0, y0, maxfev = 10000)[0]
    x4 = np.arange(0, 6, 0.01)
    y4 = f_x(x4, A, t0, C1, k1)
    plt.plot(x4, y4, "gray")
    '''

    try:
        A, B, C, D = optimize.curve_fit(func, x, y, maxfev = 100000)[0]
        x4 = np.arange(x_start, x_end, step)
        y4 = func(x4, A, B, C, D)
        plt.plot(x4, y4, "gray")
    except:
        print('fit error!')
 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return x0, [y1, y2, y3, y4], [speed_y1, speed_y2, speed_y3, speed_y4]
 
    #plt.show()


if __name__ == '__main__':
    track = np.array([[0.008000, 477.000000, 729.000000], [0.010000, 573.000000, 744.000000], [0.012000, 666.000000, 761.000000], [0.014000, 757.000000, 774.000000], [0.016000, 846.000000, 792.000000], [0.018000, 939.000000, 807.000000], [0.020000, 1027.000000, 824.000000], [0.022000, 1108.000000, 837.000000], [0.024000, 1188.000000, 849.000000], [0.026000, 1263.000000, 862.000000], [0.028000, 1341.000000, 875.000000]])
    plot_test(track[:, 0], track[:, 1], f_x)
    plot_test(track[:, 0], track[:, 2], f_y)
    for i in range(track.shape[0] - 1):
        print(track[i + 1] - track[i])