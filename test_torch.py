import torch
import numpy as np
from torch.autograd import Variable
import math



track = []
NumTime, x, y = 2e-3, 25, 37
A, B, C1, C2, t0, g = 1, 1, 1, 1, 0, 10
m = 0.005
lambda_A, lambda_B, lambda_t0 = 0.001, 0.001, 0.001
track.append((NumTime, x , y))
#tensor = torch.FloatTensor(track)
#variable = Variable(tensor, requires_grad = True)


def num2var(x):
    tensor_x = torch.FloatTensor([x])
    var_x = Variable(tensor_x, requires_grad = True)
    return var_x

v_A = num2var(A)
v_B = num2var(B)
v_C1 = num2var(C1)
v_C2 = num2var(C2)
v_t0 = num2var(t0)
f_sum = num2var(0)


for i in range(10):

    print(v_A.data, v_B.data, v_C1.data, v_C2.data, v_t0.data)

    for t, x_, y_ in track:
        f_x_t = (torch.log(v_A*(t-v_t0)+v_C1) - torch.log(v_C1))/v_A
        f_y_t_1 = torch.log(torch.exp(2*torch.sqrt(v_B*g)*(t-v_t0)+v_C2)-1)/v_B
        f_y_t_2 = torch.sqrt(g/v_B)*(t-v_t0)
        f_y_t_3 = torch.log(torch.exp(v_C2)-1)/v_B
        f_y_t = f_y_t_1-f_y_t_2-f_y_t_3
        f_sum =f_sum + (f_x_t - x_)*(f_x_t - x_) + (f_y_t - y_)*(f_y_t - y_)
        #f_sum = v_A * v_B * v_t0
        print(f_x_t, f_y_t_1, f_y_t_2, f_y_t_3)
    
    print(f_sum)
    #f_sum.backward()
    f_sum.backward(retain_graph = True)

    #print(v_A.grad, v_B.grad, v_C1.grad, v_C2.grad, v_t0.grad)
    print()

    v_A.data = v_A + lambda_A * v_A.grad.data
    v_B.data = v_B + lambda_B * v_B.grad.data
    v_C1.data = v_C1 + lambda_B * v_C1.grad.data
    v_C2.data = v_C2 + lambda_B * v_C2.grad.data
    v_t0.data = v_t0 + lambda_t0 * v_t0.grad.data


    
    v_A.grad.zero_()
    v_B.grad.zero_()
    v_C1.grad.zero_()
    v_C2.grad.zero_()
    v_t0.grad.zero_()

    

#print(v_A.data, v_B.data, v_t0.data)