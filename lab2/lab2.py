import numpy as np
import matplotlib.pyplot as plt

task = 1
var = 4


def a(x):
    if task == 1:
        return 1
    elif task == 2:
        return np.sin(var)
    elif task == 3:
        return np.sin(var*x)


def b(x):
    if task == 1:
        return 1
    elif task == 2:
        return np.cos(var)
    elif task == 3:
        return np.cos(var*x)


def p(x):
    return 1+b(x)*x**2


def f(x):
    return -1


def get_y_list(n, h, x_list, out):
    m = []
    for i in range(1, n - 1):
        row = []
        if i > 1:
            for j in range(i - 2):
                row.append(0)
            row.append(1)
        row.append(-(2 + h ** 2 * p(x_list[i]) / a(x_list[i])))
        if i < n - 2:
            row.append(1)
            for j in range(i + 1, n - 2):
                row.append(0)
        m.append(row)
    if out is True:
        print("Матрица:")
        for row in m:
            print(row)
    v = [f(x) * h ** 2 / a(x) for x in x_list[1:-1:]]
    if out is True:
        print("\nВектор:")
        for num in v:
            print(num)
    y_list = list(np.linalg.solve(m, v))
    y_list.insert(0, 0)
    y_list.append(0)
    return y_list


if __name__ == '__main__':
    n = 32
    A = -1
    B = 1
    h = (B-A)/n
    x_list = np.linspace(A, B, n)

    task = 1
    y_list = get_y_list(n, h, x_list, False)
    print("Результат:")
    for num in y_list:
        print(num)
    plt.plot(x_list, y_list, label="Задание 1", color="black")

    task = 2
    y_list = get_y_list(n, h, x_list, False)
    print("Результат:")
    for num in y_list:
        print(num)
    plt.plot(x_list, y_list, label="Задание 2", color="red")

    task = 3
    y_list = get_y_list(n, h, x_list, False)
    print("Результат:")
    for num in y_list:
        print(num)
    plt.plot(x_list, y_list, label="Задание 3", color="green")

    plt.xlabel("x")
    plt.xlabel("y")
    plt.legend()
    plt.grid()
    plt.show()
