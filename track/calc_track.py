import numpy as np
from scipy import optimize
 
#直线方程函数
def f_1(x, A, B):
    return A*x + B
 
#二次曲线方程
def f_2(x, A, B, C):
    return A*x*x + B*x + C
 
#三次曲线方程
def f_3(x, A, B, C, D):
    return A*x*x*x + B*x*x + C*x + D

#四次曲线方程
def f_4(x, A, B, C, D, E):
    return A*x*x*x*x + B*x*x*x + C*x*x + D*x + E

#五次曲线方程
def f_5(x, A, B, C, D, E, F):
    return A*x*x*x*x*x + B*x*x*x*x + C*x*x*x + D*x*x + E*x + F

def f_poly(x_list, params):
    result = np.zeros_like(x_list, dtype = np.float)
    for i in range(len(params)):
        result = result * x_list + params[i]
    return result

def create_curve(t_start, t_end, n_step, params):
    step_size = (t_end - t_start) / n_step 
    t_list = np.arange(t_start, t_end, step_size)
    curve = []
    for i in range(len(params)):
        curve.append(f_poly(t_list, params[i]))
    return curve


def fitting(t, coord, poly_time = 3):
    func_list = [f_1, f_2, f_3, f_4, f_5]
    
    params = [(), (), ()]
    curve = [[], [], []]
    func = func_list[poly_time - 1]
    
    error = np.zeros_like(t, dtype = np.float)
             
    for i in range(3):
        params[i] = optimize.curve_fit(func, t, coord[i])[0]
        fit_curve = f_poly(t, params[i])
        error += np.abs(coord[i] - fit_curve)
        
    return params, error

def calc_dis(coor0, coor1):
    ans = 0
    for i in range(len(coor0)):
        ans += (coor0[i] - coor1[i]) ** 2
    return ans ** 0.5

def get_parent(no):
    global parent
    if parent[no] != no:
        parent[no] = get_parent(parent[no])
    return parent[no]

def judge_xy(times, coor_xs, coor_ys):
    pairs = [(times[i], coor_xs[i], coor_ys[i]) for i in range(len(times))]
    pairs.sort(key = lambda x : x[0])
    for k in range(1, 3):
        trend = 0
        for i in range(len(pairs) - 1):
            for j in range(i + 1, len(pairs)):
                if abs(pairs[i][k] - pairs[j][k]) > 5:
                    if pairs[i][k] > pairs[j][k]:
                        if trend > 0:
                            return False
                        else:
                            trend = -1
                    else:
                        if trend < 0:
                            return False
                        else:
                            trend = 1
    return True
    
def fitting_kruskal(coordinary_list):
    global parent

    try_set = set()
    threshold = 60
    time, x, y, z = coordinary_list[:, 0] * 0.01, coordinary_list[:, 1], coordinary_list[:, 2], coordinary_list[:, 3]

    parent = [no for no in range(len(time))]
    candidate_traces = [[no] for no in range(len(time))]
    
    for t in range(1, 21):
        print('t=', t)
        dis_list = []
        for i in range(len(time) - 1):
            for j in range(i + 1, len(time)):
                if time[i] == time[j] or get_parent(i) == get_parent(j) or time[i] + 0.05 * (t-1) >= time[j]:
                    continue
                if time[i] + 0.05 * t < time[j]:
                    break
                dis_list.append([calc_dis(coordinary_list[i, 1 : 4], coordinary_list[j, 1 : 4]), i, j])
        dis_list.sort(key = lambda x: x[0])
        print('len(dis_list) = ', len(dis_list))
        for num, (dis, i, j) in enumerate(dis_list):
            if num % 1000 == 0:
                print(num)
            p_i, p_j = get_parent(i), get_parent(j)
            if p_i == p_j or (p_i, p_j) in try_set or (p_j, p_i) in try_set:
                continue
            try_set.add((p_i, p_j))
            trace = candidate_traces[p_i].copy()
            trace.extend(candidate_traces[p_j])
            if not judge_xy(time[trace], x[trace], y[trace]):
                continue
            if len(trace) > 5:
                _, error = fitting(time[trace], (x[trace], y[trace], z[trace]), 2)
                if error.max() > threshold:
                    continue
            candidate_traces[p_i] = trace
            parent[p_j] = p_i
        
    return candidate_traces


class fit_curve():
    def __init__(self, time, coord):
        self.fit_threshold, self.speed_threshold = 15, 300 / 3.6 * 100 
        self.time, self.coord = time, coord
        self.direction = np.sign(coord[0][-1] - coord[0][0])
        self.fitting()
        #self.fitting_kruskal()
        self.start, self.end = self.time.min(), self.time.max()
    
    def create_curve(self, t_start = None, t_end = None, n_step= None):
        if t_start is None or t_end is None or n_step is None:
            t_start, t_end, n_step = self.start, self.end, 1000. 
        curve = create_curve(t_start, t_end, n_step, self.params)  
        x_dir, y_dir, z_dir = np.sign(curve[0][1] - curve[0][0]), np.sign(curve[1][1] - curve[1][0]), np.sign(curve[2][1] - curve[2][0])
        
        for i in range(1, len(curve[0]) - 1):
            if x_dir * (curve[0][i + 1] - curve[0][i]) < 0 or y_dir * (curve[1][i + 1] - curve[1][i]) < 0 or (z_dir < 0 and curve[2][i + 1] > curve[2][i]):
                for j in range(len(curve)):
                    curve[j] = curve[j][:i + 1]
                break
        
        return curve
    '''
    def get_parent(self, no):
        if self.parent[no] != no:
            self.parent[no] = self.get_parent(self.parent[no])
        return self.parent[no]
        
    
    def fitting_kruskal(self):
        time, x, y, z = self.time, self.coord[0], self.coord[1], self.coord[2]
        
        
        params, error = fitting(time, (x, y, z), 2)
        print('max error:', error.max())
        if error.max() <= 3 * self.fit_threshold:
            self.params = params
            return
        
        boo = [[True for _ in range(len(time))] for _ in range(len(time))]
        self.parent = [no for no in range(len(time))]
        candidate_traces = [[no] for no in range(len(time))]
        dis_list = [(calc_dis((x[i], y[i], z[i]), (x[j], y[j], z[j])), i, j) for i in range(len(time - 1)) for j in range(i + 1, len(time))]
        dis_list.sort(key = lambda x: x[0])
        for (dis, i, j) in dis_list:
            p_i, p_j = self.get_parent(i), self.get_parent(j)
            if p_i == p_j or not boo[p_i][p_j]:
                continue
            boo[p_i][p_j] = boo[p_j][p_i] = False
            trace = candidate_traces[p_i].copy()
            trace.extend(candidate_traces[p_j])
            if len(trace) > 5:
                _, error = fitting(time[trace], (x[trace], y[trace], z[trace]))
                if error.max() > self.fit_threshold * 3:
                    continue
            candidate_traces[p_i] = trace
            self.parent[p_j] = p_i
        
        candidate_traces.sort(key = lambda x : len(x), reverse = True)
        final_trace = candidate_traces[0]
        #final_trace = [no for no in range(len(time))]
        self.params, error =  fitting(time[final_trace], (x[final_trace], y[final_trace], z[final_trace]))
        final_trace.sort()
        self.time, self.coord = time[final_trace], (x[final_trace], y[final_trace], z[final_trace])
        print('mean error:', error.sum() / (3 * len(error)))
    '''   
    
    def fitting(self):
        time, x, y, z = self.time.copy(), self.coord[0].copy(), self.coord[1].copy(), self.coord[2].copy()
        candidate_traces = []    
        
        '''
        for i in range(len(time)):
            in_trace = False
            for num, trace in enumerate(candidate_traces):
                index = trace[-1]
                if num == 0:
                    print(time[i], x[i], y[i], z[i])
                    print(calc_dis((x[index], y[index], z[index]), (x[i], y[i], z[i])), (time[i] - time[index]) * self.speed_threshold)
                if calc_dis((x[index], y[index], z[index]), (x[i], y[i], z[i])) < (time[i] - time[index]) * self.speed_threshold:
                    trace.append(i)
                    in_trace = True
            if not in_trace:
                candidate_traces.append([i])
        
        final_trace = []
        for trace in candidate_traces:
            if len(final_trace) < len(trace):
                final_trace = trace
        print(final_trace)
        time, x, y, z = time[final_trace], x[final_trace], y[final_trace], z[final_trace]
        '''
        
        self.params, error = fitting(time, (x, y, z), 3)
        while len(time) > 10:
            self.params, error = fitting(time, (x, y, z), 3)
            x_fit, y_fit, z_fit = f_poly(time, self.params[0]), f_poly(time, self.params[1]), f_poly(time, self.params[2])
            mean_error = error.sum() / (3 * len(time))
            #print(error.max())
            if error.max() < self.fit_threshold:
                break
            index = error.argmax()
            time, x, y, z = np.delete(time, index), np.delete(x, index), np.delete(y, index), np.delete(z, index)


    
    @staticmethod
    def calc_poly(t, params):
        ans = 0
        for i in range(len(params)):
            ans = ans * t + params[i]
        return ans
    @staticmethod
    def calc_poly_derivative(t, params):
        ans = 0
        for i in range(len(params) - 1):
            time = len(params) - i - 1
            ans += time * params[i] * (t ** (time - 1)) 
        return ans


    def calc_droppoint(self):
        drop_time = optimize.fsolve(self.calc_poly, self.time[-1], args = (self.params[2]))[0]
        x, y, z = self.calc_poly(drop_time, self.params[0]), self.calc_poly(drop_time, self.params[1]), self.calc_poly(drop_time, self.params[2])
        
        if False and abs(z) < 1e-2:
            return x, y
        else:
            x, y, z = self.coord
            for i in range(-13, -9):
                tmp_params, _ = fitting(self.time[i :], (x[i :], y[i :], z[i :]), 2)
                drop_time = optimize.fsolve(self.calc_poly, self.time[-1], args = (tmp_params[2]))[0]
                drop_x, drop_y, drop_z = self.calc_poly(drop_time, tmp_params[0]), self.calc_poly(drop_time, tmp_params[1]), self.calc_poly(drop_time, tmp_params[2])
                if abs(drop_z) < 1e-2:
                    return drop_x, drop_y
            return None
    
    def calc_startpoint(self):
        x, y, z = self.coord
        tmp_params, _ = fitting(self.time[: 5], (x[: 5], y[: 5], z[: 5]), 1)
        start_time = optimize.fsolve(self.calc_poly, self.time[-1], args = (tmp_params[2]))[0]
        if start_time <= self.time[0]:
            x, y = self.calc_poly(start_time, tmp_params[0]), self.calc_poly(start_time, tmp_params[1])
            return x, y
        else:
            return None

    def calc_speed(self):
        drop_speed, pass_speed = -1, -1
        
        drop_time = optimize.fsolve(self.calc_poly, self.time[-1], args = (self.params[2]))[0]
        z = self.calc_poly(drop_time, self.params[2])
    
        if False and abs(z) < 1e-2:
            drop_speed = 0
            for i in range(3):
                drop_speed += self.calc_poly_derivative(drop_time, self.params[i]) ** 2
            drop_speed = drop_speed ** 0.5
        else:
            x, y, z = self.coord
            for i in range(-13, -9):
                tmp_params, _ = fitting(self.time[i :], (x[i :], y[i :], z[i :]), 2)
                drop_time = optimize.fsolve(self.calc_poly, self.time[-1], args = (tmp_params[2]))[0]
                drop_z = self.calc_poly(drop_time, tmp_params[2])
                if abs(drop_z) < 1e-2:
                    drop_speed = 0
                    for i in range(3):
                        drop_speed += self.calc_poly_derivative(drop_time,  tmp_params[i]) ** 2
                    drop_speed = drop_speed ** 0.5
                    break

        pass_time = optimize.fsolve(self.calc_poly, (self.time[-1] + self.time[0]) / 2, args = (self.params[0]))[0]
        x = self.calc_poly(pass_time, self.params[0])
    
        if abs(x) < 1e-2:
            pass_speed = 0
            for i in range(3):
                pass_speed += self.calc_poly_derivative(pass_time, self.params[i]) ** 2
            pass_speed = pass_speed ** 0.5
        
        return drop_speed, pass_speed
    
    def classify(self):
        #目前将杀球与吊球统一为下压，推球统一到平抽挡中
        self.type_list = ['高远', '下压', '挑球', '网前', '抽挡', '未知']
        self.type = self.type_list.index('未知')
        
        curve = self.create_curve()
        
        max_high = curve[2].max()
        if max_high > 300:
            #高远 挑球
            if abs(curve[0][0]) > 335 and curve[0][0] * curve[0][-1] < 0 and curve[2][0] > 170:
                self.type = self.type_list.index('高远')
            elif curve[2][0] <= 170:
                self.type = self.type_list.index('挑球')
            else:
                ret = self.calc_startpoint()
                if ret is not None:
                    x, y = ret
                    if abs(x) > 590 and x * curve[0][-1] < 0:
                        self.type = self.type_list.index('高远')
                    else:
                        self.type = self.type_list.index('挑球')
                else:
                    self.type = self.type_list.index('未知')
        else:
            #下压 放网 抽挡
            if (True or abs(curve[0][0]) > 335) and  curve[2][0] > 200:
                self.type = self.type_list.index('下压')
            else:
                #放网 抽挡
                ret = self.calc_droppoint()
                if ret is not None:
                    x, y = ret
                    if abs(x) < (198 + 5) and abs(curve[0][0]) < (198 + 5):
                        self.type = self.type_list.index('网前')         
                    else:
                        self.type = self.type_list.index('抽挡')
                else:
                    self.type = self.type_list.index('未知')

def poly(x, params):  
    ans = np.zeros(len(params), dtype = np.float)
    for i in range(len(params)):
        for j in range(len(params[i])):
            ans[i] = ans[i] * x + params[i][j]
    return ans

def residuals(t, params0, params1):
    return poly(t, params0) - poly(t, params1)

def intersect(t_start, t_end, params0, params1):
    t_result, _ = optimize.leastsq(residuals, (t_start + t_end) / 2, args = (params0, params1))
    t_result = min(t_end, max(t_start, t_result[0]))
    return t_result

def dp(coordinary_list, candidate_list, traces):
    global parent

    f = [(0, -1) for _ in range(len(coordinary_list) + 1)]
    no = 0

    for i in range(len(coordinary_list)):
        if f[i][0] > f[i + 1][0]:
            f[i + 1] = (f[i][0], -1)
        if no >= len(candidate_list):
            continue
        while no < len(candidate_list) and candidate_list[no][0] == i:
            if candidate_list[no][2] > 4 and f[i][0] + candidate_list[no][2] > f[candidate_list[no][1] + 1][0]:
                f[candidate_list[no][1] + 1] = (f[i][0] + candidate_list[no][2], no) 
            no += 1

    chosen_list = []
    t = len(coordinary_list)
    while t > 0:
        if f[t][1] == -1:
            t = t - 1
        else:
            chosen_list.append(f[t][1])
            t = candidate_list[f[t][1]][0]
            
    candidate_list2 = []

    for i in range(len(parent)):
        if parent[i] == i:
            tmp_trace = np.array(traces[i])
            candidate_list2.append(traces[i])
            
    chosen_list.reverse()

    traces = []
    for index in chosen_list:
        traces.extend(candidate_list2[index])
    traces.sort()

    coordinary_list = coordinary_list[traces]
    return coordinary_list

def calc_track(track_path):
    global parent
    with open(track_path) as f:
        data = f.read().split('\n')[:-1]
    coordinary_list = []

    for term in data:
        new_term = term.split(' ')[:-1]
        coordinary = [float(x) for x in new_term]
        if True or len(coordinary_list) == 0 or coordinary_list[-1][0] < coordinary[0]:
            coordinary_list.append(coordinary)
        
        '''
        if coordinary[0] > 1000:
            break
        '''
        
        

    coordinary_list = np.array(coordinary_list)
    traces = fitting_kruskal(coordinary_list)

    candidate_list = []
    for i in range(len(parent)):
        if parent[i] == i:
            tmp_trace = np.array(traces[i])
            candidate_list.append([tmp_trace.min(), tmp_trace.max(), len(tmp_trace)])

    candidate_list.sort(key = lambda x: x[0])

    '''
    f = [(0, -1) for _ in range(len(coordinary_list) + 1)]

    no = 0

    for i in range(len(coordinary_list)):
        if f[i][0] > f[i + 1][0]:
            f[i + 1] = (f[i][0], -1)
        if no >= len(candidate_list):
            continue
        while no < len(candidate_list) and candidate_list[no][0] == i:
            if candidate_list[no][2] > 4 and f[i][0] + candidate_list[no][2] > f[candidate_list[no][1] + 1][0]:
                f[candidate_list[no][1] + 1] = (f[i][0] + candidate_list[no][2], no) 
            no += 1

    chosen_list = []
    t = len(coordinary_list)
    while t > 0:
        if f[t][1] == -1:
            t = t - 1
        else:
            chosen_list.append(f[t][1])
            t = candidate_list[f[t][1]][0]
            
    candidate_list2 = []

    for i in range(len(parent)):
        if parent[i] == i:
            tmp_trace = np.array(traces[i])
            candidate_list2.append(traces[i])
            
    chosen_list.reverse()

    traces = []
    for index in chosen_list:
        traces.extend(candidate_list2[index])
    traces.sort()

    coordinary_list = coordinary_list[traces]
    '''
    coordinary_list = dp(coordinary_list, candidate_list, traces)
    if coordinary_list[0, 1] < coordinary_list[1, 1]:
        x_dir = 1
    else:
        x_dir = -1
        
    hit_list = [0]

    for i in range(2, len(coordinary_list)):
        if (coordinary_list[i][1] - coordinary_list[i - 1][1]) * x_dir < -6:
            hit_list.append(i)
            x_dir = -x_dir
            #print('hit between frame %d %d'%(coordinary_list[i - 1][0], coordinary_list[i][0]))
    hit_list.append(len(coordinary_list))

    curve_list = []


    for i in range(len(hit_list) - 1):
        segment = coordinary_list[hit_list[i] : hit_list[i + 1] - 1]
        time, x, y, z = np.array([t[0] * 0.01 for t in segment]), np.array([t[1] for t in segment]), np.array([t[2] for t in segment]), np.array([t[3] for t in segment])
        for j in range(len(z)):
            if z[j] < 7:
                time, x, y, z = time[:j], x[:j], y[:j], z[:j]
                break
        
        if len(time) < 10:
            continue
                
    
        
        poly_curve = fit_curve(time, (x, y, z))
        if poly_curve.params is None:
            continue
        curve_list.append(poly_curve)
        
        
        if len(curve_list) > 1:
            print(curve_list[-2].end, curve_list[-1].start)
            if curve_list[-2].end + 0.5 > curve_list[-1].start:
                hit_time = intersect(curve_list[-2].end, curve_list[-1].start, curve_list[-2].params, curve_list[-1].params)
                #curve_list[-1].start = curve_list[-2].end = hit_time
                curve_list[-1].start = hit_time
                print(curve_list[-2].end, curve_list[-1].start, hit_time)

        
    for i, curve in enumerate(curve_list):
        print(i + 1)
        curve.classify()
    return curve_list





