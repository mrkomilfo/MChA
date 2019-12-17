import numpy as np
import matplotlib.pyplot as plt

'''
L = 25
delta_u = 0.2
E = 120e9
rho = 7.4

x_num = 75
x_list = np.linspace(0, L, x_num)
h = L / x_num

T = 1
tau = h / np.sqrt(E/rho)
t_num = int(T / tau)

#p = lambda x: abs(x-L/2)-L/2
#p = lambda x: np.sqrt((L/2)**2-(x-L/2)**2)
#p = lambda x: np.sin(x*np.pi/L)
p = lambda x: 0.04*x**2 - x
f = lambda x: 0
q = lambda x: 0


def solve():
    M = np.zeros((t_num, x_num), dtype=np.float)
    M[0, 1:-1] = p(x_list[1:-1])
    M[1, 1:-1] = p(x_list[1:-1]) + tau * q(x_list[1:-1]) + tau**2 / 2 * (f(x_list[1:-1]) + p(x_list[2:]) - 2 * p(x_list[1:-1]) + p(x_list[:-2]))

    expr = E/rho * tau**2 / h**2
    for i in range(2, t_num):
        M[i, 1:-1] = -M[i-2, 1:-1] + expr * (M[i-1, 2:] + M[i-1, :-2]) + M[i-1, 1:-1] * (2 - 2*expr)
    return M


def draw_plot(u, x_list):
    for u_x in [u[:][i] for i in range(u.shape[1])]:
        plt.plot(x_list, u_x)
    plt.grid()
    plt.show()
    

if __name__ == '__main__':
    M = solve()
    draw_plot(M, x_list)
'''

a = 3
b = 1

x_num = 40
x_h = a / x_num
x_list = np.linspace(-a / 2, a / 2, x_num)

y_num = 40
y_h = b / y_num
y_list = np.linspace(-b / 2, b / 2, y_num)

T = 10
tau = x_h * y_h / np.sqrt(x_h ** 2 + y_h ** 2)
t_num = int(T / tau)

u_0 = lambda x, y: 2 * np.cos(np.pi * x / a)
du_dt_0 = lambda x, y: np.tan(np.sin(2*np.pi * x / a)) * np.sin(np.pi * y / b)


def solve_2():
    M = np.zeros((x_num, y_num, t_num))
    u_0_matrix = np.zeros((x_num, y_num))
    du_dt_0_matrix = np.zeros((x_num, y_num))

    for i in range(x_num):
        for j in range(y_num):
            u_0_matrix[i, j] = u_0(x_list[i], y_list[j])
            du_dt_0_matrix[i, j] = du_dt_0(x_list[i], y_list[j])

    M[1:-1, :, 0] = u_0_matrix[1:-1]

    for i in range(1, x_num - 1):
        for j in range(1, y_num - 1):
            M[i, j, 1] = u_0_matrix[i, j] + tau * du_dt_0_matrix[i, j] + tau ** 2 / 2 * \
                         ((u_0_matrix[i + 1, j] - 2 * u_0_matrix[i, j] + u_0_matrix[i - 1, j]) / x_h ** 2 +
                          (u_0_matrix[i, j + 1] - 2 * u_0_matrix[i, j] + u_0_matrix[i, j - 1]) / y_h ** 2)

    M[1:x_num-1, 0, 1] = M[1:x_num-1, 1, 1]
    M[1:x_num-1, -1, 1] = M[1:x_num-1, -2, 1]

    for k in range(2, t_num):
        for i in range(1, x_num - 1):
            for j in range(1, y_num - 1):
                M[i, j, k] = 2 * M[i, j, k - 1] * (1 - (tau / x_h) ** 2 - (tau / y_h) ** 2) + \
                             (M[i, j + 1, k - 1] + M[i, j - 1, k - 1]) * (tau / y_h) ** 2 + \
                             (tau / x_h) ** 2 * (M[i + 1, j, k - 1] + M[i - 1, j, k - 1]) - M[i, j, k-2]
            M[i, 0, k] = M[i, 1, k]
            M[i, -1, k] = M[i, -2, k]

    return M


def x_plot(M, x, h):
    for s in M[x, :, ::h].transpose():
        plt.plot(x_list, s)
    plt.title("Проекция x")
    plt.grid()
    plt.show()


def y_plot(M, y, h):
    for s in M[:, y, ::h].transpose():
        plt.plot(y_list, s)
    plt.title("Проекция y")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    M = solve_2()
    x_plot(M, 1, 20)
    y_plot(M, 1, 20)
