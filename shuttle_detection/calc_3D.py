import numpy as np

def calc_3D(mats, points):
    A = np.zeros((4, 3), dtype = np.float)
    b = np.zeros(4, dtype = np.float)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                A[2 * i + j, k] = points[i][j] * mats[i][2, k] - mats[i][j, k]
            b[2 * i + j] = mats[i][j, 3] - points[i][j] * mats[i][2, 3]
    
    A = np.mat(A)
    return np.dot(np.dot((np.dot(A.T, A)).I, A.T), b)