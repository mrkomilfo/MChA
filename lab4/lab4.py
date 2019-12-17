from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 2
k = 1
T = 0.2

x_num_default = 50
h_default = (b - a) / x_num_default
x_list_default = np.linspace(a, b, x_num_default)

tau_default = 0.5 * (h_default**2 / k)
t_num_default = int(T / tau_default)
t_list_default = np.linspace(0, T, t_num_default)

phi = lambda x: 0
g1 = lambda t: 0
g2 = lambda t: 0
f = lambda x, t: x


def draw_plot(M, num, title):
    t_selection = np.linspace(0, t_num_default - 1, num)
    for t in t_selection:
        plt.plot(x_list_default, M[int(t)])
    plt.title(title)
    plt.grid()
    plt.show()


def explicit(method, h, tau):
    x_num = int((b - a) / h)
    x_list = np.linspace(a, b, x_num)
    t_num = int(T / tau)
    t_list = np.linspace(0, T, t_num)

    M = np.zeros((t_num, x_num))
    M[0, :] = [phi(x_i) for x_i in x_list]
    M[:, 0] = [g1(t_i) for t_i in t_list]

    for i in range(1, t_num):
        for j in range(1, x_num - 1):
            M[i, j] = tau * M[i - 1, j - 1] / h ** 2 + (1 - 2 * tau / h ** 2) * M[i - 1, j] + \
                      tau / h ** 2 * M[i - 1, j + 1] + tau * f(x_list[j], t_list[i - 1])

        M[i, -1] = g2(t_list[i]) * h + M[i, -2] if method == 1 else \
            tau * M[i - 1, -2] / h ** 2 + (1 - 2 * tau / h ** 2) * M[i - 1, -1] + \
            tau / h ** 2 * (2 * h * g2(t_list[i]) + M[i, -2]) + tau * f(x_list[-1], t_list[i - 1])
    return M


def implicit(method, h, tau):
    t_num = int(T / tau)
    t_list = np.linspace(0, T, t_num)
    x_num = int((b - a) / h)
    x_list = np.linspace(a, b, x_num)

    M = np.zeros((t_num, x_num), dtype=np.float)
    M[0, :] = [phi(x) for x in x_list]

    for i in range(1, t_num):
        U, V = (np.zeros((x_num, x_num)), np.zeros(x_num)) if method == 1 else \
            (np.zeros((x_num+1, x_num+1)), np.zeros(x_num+1))
        for j in range(1, V.shape[0] - 1):
            U[j, j-1:j+2] = -k / h ** 2, 1 / tau + 2 * k / h ** 2, -k / h ** 2

        U[0, 0] = 1
        U[-1, -1] = 1
        U[-1, -2 if method == 1 else -3] = -1

        V[0] = g1(x_list[0])
        for j in range(1, V.shape[0] - 1):
            V[j] = f(x_list[j], t_list[i]) + M[i - 1, j] / tau

        V[-1], M[i] = (h * g2(t_list[i]), np.linalg.solve(U, V)) if method == 1 else \
            (2 * h * g2(t_list[i]), np.linalg.solve(U, V)[:-1])

    return M


def table_for_fixed_h(scheme, method, h, tau, n=5, precision=8):
    x_num = int((b - a) / h)
    t_num = int(T / tau)
    t_list = np.linspace(0, T, t_num)

    tn_1_index, tn_2_index = np.random.randint(0, t_num-1, size=2)
    tn_1, tn_2 = t_list[tn_1_index], t_list[tn_2_index]
    print("tn1 = %f, tn2 = %.8f " % (tn_1, tn_2))
    table = PrettyTable()
    table.field_names = ['N', 'τ', 's(tn1)', 's(tn2)', 'max(t1)', 'max(t2)']
    for i in range(0, n):
        tn_1_index, tn_2_index = int(tn_1 / tau), int(tn_2 / tau)

        M_1 = np.matrix('0')
        M_2 = np.matrix('0')
        if scheme == 1:
            M_1 = explicit(method, h, tau)
            M_2 = explicit(method, h, tau / 2)
        elif scheme == 2:
            M_1 = implicit(method, h, tau)
            M_2 = implicit(method, h, tau / 2)

        diff_tn_1, diff_tn_2 = M_1[tn_1_index] - M_2[2 * tn_1_index], M_1[tn_2_index] - M_2[2 * tn_2_index]
        std_tn_1, std_tn_2 = diff_tn_1.std(), diff_tn_2.std()
        max_tn_1, max_tn_2 = max(diff_tn_1), max(diff_tn_2)
        table.add_row([x_num, tau, round(std_tn_1, precision), round(std_tn_2, precision),
                       round(max_tn_1, precision), round(max_tn_2, precision)])
        tau /= 2
    print(table)


def table_for_fixed_tau(scheme, method, h, tau, n=5, precision=8):
    t_num = int(T / tau)
    t_list = np.linspace(0, T, t_num)
    x_num = int((b - a) / h)

    tn_1_index, tn_2_index = np.random.randint(0, t_num-1, size=2)
    tn_1, tn_2 = t_list[tn_1_index], t_list[tn_2_index]
    print("tn1 = %f, tn2 = %.8f " % (tn_1, tn_2))

    table = PrettyTable()
    table.field_names = ['N', 'h', 's(tn1)', 's(tn2)', 'max(t1)', 'max(t2)']
    for i in range(0, n):
        tn_1_index, tn_2_index = int(tn_1 / tau), int(tn_2 / tau)

        M_1 = np.matrix('0')
        M_2 = np.matrix('0')
        if scheme == 1:
            M_1 = explicit(method, h, tau)
            M_2 = explicit(method, h / 2, tau)
        elif scheme == 2:
            M_1 = implicit(method, h, tau)
            M_2 = implicit(method, h / 2, tau)

        diff_tn_1, diff_tn_2 = M_1[tn_1_index] - M_2[tn_1_index, 1::2], M_1[tn_2_index] - M_2[tn_2_index, 1::2]
        std_tn_1, std_tn_2 = diff_tn_1.std(), diff_tn_2.std()
        max_tn_1, max_tn_2 = max(diff_tn_1), max(diff_tn_2)
        table.add_row([x_num, h, round(std_tn_1, precision), round(std_tn_2, precision),
                       round(max_tn_1, precision), round(max_tn_2, precision)])
        h /= 2
        x_num = int((b - a) / h)
    print(table)


if __name__ == "__main__":
    M = explicit(1, h_default, tau_default)
    draw_plot(M, 16, "Явная разностная схема \nСпособ 1")
    table_for_fixed_h(1, 1, h_default, h_default**2/2)
    table_for_fixed_tau(1, 1, 0.5, 0.001)

    M = explicit(2, h_default, tau_default)
    draw_plot(M, 16, "Явная разностная схема \nСпособ 2")
    table_for_fixed_h(1, 2, h_default, h_default**2/2)
    table_for_fixed_tau(1, 2, 0.5, 0.001)

    M = implicit(1, h_default, tau_default)
    draw_plot(M, 16, "Неявная разностная схема \nСпособ 1")
    table_for_fixed_h(2, 1, h_default, h_default**2, precision=10)
    table_for_fixed_tau(2, 1, 0.1, 0.01)

    M = implicit(2, h_default, tau_default)
    draw_plot(M, 16, "Неявная разностная схема \nСпособ 2")
    table_for_fixed_h(2, 2, h_default, h_default**2, precision=10)
    table_for_fixed_tau(2, 2, 0.1, 0.01)
