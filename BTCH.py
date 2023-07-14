"""
    Function：Bayesian Three-Cornered Hat (BTCH) Method
    Author：Tongren Xu (xutr@bnu.edu.cn), Xinlei He (hxlbsd@mail.bnu.edu.cn) and Gangqiang Zhang(zhanggq@mail.bnu.edu.cn)
    Version：Python 3.6.8
    Reference: Xu, T., Guo, Z., Xia, Y., et al. 2019. Evaluation of twelve evapotranspiration products from machine learning, remote sensing and land surface models over conterminous United States. Journal of Hydrology 578, 124105. https://doi.org/10.1016/j.jhydrol.2019.124105
    Reference: He, X., Xu, T., Xia, Y., et al. 2020. A Bayesian Three-Cornered Hat (BTCH) Method: Improving the Terrestrial Evapotranspiration Estimation. Remote Sensing 12, 878. https://doi.org/10.3390/rs12050878
"""
import numpy as np
from scipy.optimize import minimize


def tch(x):
    # objective function
    def fun_object(r, S):
        N = S.shape[0]
        f = 0.0
        K2 = np.linalg.det(S) ** (2 / N)
        for i in range(N):
            f = f + r[i] ** 2  # why?
            for j in range(i + 1, N):
                f = f + (S[i, j] - r[N] + r[i] + r[j]) ** 2
        return f / K2

    # constraint conditions
    def fun_constraint(r, S):
        N = S.shape[0]
        u = np.full((N,), 1.0)
        K = np.linalg.det(S) ** (1 / N)
        f = (r[N] - (r[:-1] - r[N] * u).dot(np.linalg.inv(S)).dot(r[:-1] - r[N] * u)) / K
        return f

    def tch(x):
        M, N = x.shape  # M: samples; N:variables
        N_ref = N  # the last column as the reference dataset
        y_list = []
        for i in range(N):
            if i == N_ref - 1:
                pass
            else:
                y_list.append(x[:, i] - x[:, N_ref - 1])
        Y = np.vstack(y_list).T  # size = M*N-1
        S = np.cov(Y.T)  # it is different from the operation of the matlab cov. size:(N-1 * N-1)
        u = np.full((1, N - 1), 1.0)
        R = np.zeros((N, N))
        R[N - 1, N - 1] = 1 / (2 * u.dot(np.linalg.inv(S)).dot(u.T))
        X0 = R[:, N - 1]
        # According to the initial conditions, constraint conditions, and objective function of the iteration, R(:,N-1) is calculated
        res = minimize(lambda r: fun_object(r, S), X0, method='SLSQP', constraints={'type': 'ineq', 'fun': lambda r: fun_constraint(r, S)})
        R[:, N - 1] = res.x
        R[0:-1, 0:-1] = S - R[N - 1, N - 1] * (u.T.dot(u)) + u.T.dot(R[:-1, N - 1:N].T) + R[:-1, N - 1:N].dot(u)
        std = np.round([np.sqrt(R[ii, ii]) for ii in range(N)], 3)
        std_xd = np.round([np.sqrt(R[ii, ii]) / np.nanmean(abs(x[:, ii])) for ii in range(N)], 3)
        print("std:{}\nstd_xd:{}".format(std, std_xd))
        return std, std_xd

    return tch(x)


def btch(all_x):
    N, M, R, C = all_x.shape  # N:number of products, M:samples, R:rows, C:columns
    STD, T_F = np.full((N, R, C), np.nan), np.full((R, C), np.nan)
    all_x_nan_rc, all_x_nan_mean = [], []
    for r in range(R):
        for c in range(C):
            if np.isnan(all_x[:, :, r, c]).sum() == 0:
                x = all_x[:, :, r, c].T
                TCH_return = tch(x)
                if np.isnan(TCH_return[0]).sum() == 0 and (TCH_return[0] == 0.).sum() == 0:
                    STD[:, r, c], T_F[r, c] = TCH_return[0] ** 2, TCH_return[2]
                else:
                    T_F[r, c] = False
                print("R:{:03d}, C:{:03d}".format(r, c), T_F[r, c])
            else:
                all_x_nan_rc.append([r, c])
                all_x_nan_mean.append(np.nanmean(all_x[:, :, r, c].T, axis=1))

    STD_all = np.full((R, C), 0.0)
    for i in range(N):
        STD_all = STD_all + np.prod(STD, axis=0) / STD[i]

    Wight_all = np.full((N, R, C), 0.0)
    for i in range(N):
        Wight_all[i] = np.prod(STD, axis=0) / STD[i] / STD_all

    all_x_btch = np.full((M, R, C), 0.0)
    for i in range(N):
        all_x_btch = all_x_btch + all_x[i] * Wight_all[i]

    F_R, F_C = np.where(T_F == 0)

    for i in range(len(F_R)):
        all_x_btch[:, F_R[i], F_C[i]] = all_x[0, :, F_R[i], F_C[i]].T

    for i in range(len(all_x_nan_rc)):
        all_x_btch[:, all_x_nan_rc[i][0], all_x_nan_rc[i][1]] = all_x[0, :, all_x_nan_rc[i][0], all_x_nan_rc[i][1]].T
    return all_x_btch


if __name__ == '__main__':
    # Get datasets based on own projects
    # all_x: (shape:N*M*R*C; N:number of products, M:samples, R:rows, C:columns)
    all_x = None

    # BTCH results
    all_x_btch = btch(all_x)
