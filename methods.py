import numpy as np
from typing import Callable

"""
Функции для численного решения систем обыкновенных дифференциальных уравнений.

Все функции принимают следующие параметры:
    f (Callable[[list[float], list[float]], list[float]]): Функция, задающая правую часть системы ОДУ.
    y0 (float): Начальное условие для решения.
    t0 (float): Начальное значение независимой переменной.
    t1 (float): Конечное значение независимой переменной.
    h (float): Шаг интегрирования.

Все функции возвращают кортеж (tuple[np.ndarray, np.ndarray]), содержащий массивы
со значениями независимой переменной и решения системы ОДУ.
"""


def euler_method(
    f: Callable[[list[float], list[float]], list[float]],
    y0: float,
    t0: float,
    t1: float,
    h: float,
):
    """Метод Эйлера"""

    t_values = np.arange(t0, t1, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        y_values[i] = y_values[i - 1] + h * f(y_values[i - 1], t_values[i - 1])
    return t_values, y_values


def euler_modified_method(
    f: Callable[[list[float], list[float]], list[float]],
    y0: float,
    t0: float,
    t1: float,
    h: float,
):
    """Метод Эйлера с пересчетом"""

    t_values = np.arange(t0, t1, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        y_pred = y_values[i - 1] + h * f(y_values[i - 1], t_values[i - 1])
        y_values[i] = y_values[i - 1] + 0.5 * h * (
            f(y_values[i - 1], t_values[i - 1]) + f(y_pred, t_values[i - 1] + h)
        )
    return t_values, y_values


def runge_kutta_4th_order(
    f: Callable[[list[float], list[float]], list[float]],
    y0: float,
    t0: float,
    t1: float,
    h: float,
):
    """Метод Рунге-Кутты 4 порядка"""

    t_values = np.arange(t0, t1, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        k1 = h * f(y_values[i - 1], t_values[i - 1])
        k2 = h * f(y_values[i - 1] + k1 / 2, t_values[i - 1] + h / 2)
        k3 = h * f(y_values[i - 1] + k2 / 2, t_values[i - 1] + h / 2)
        k4 = h * f(y_values[i - 1] + k3, t_values[i - 1] + h)

        y_values[i] = y_values[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values


def adams_bashforth_4th_order(
    f: Callable[[list[float], list[float]], list[float]],
    y0: float,
    t0: float,
    t1: float,
    h: float,
):
    """Метод Адамса-Бешфорта 4 порядка"""

    t_values = np.arange(t0, t1, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    t_rk, y_rk = runge_kutta_4th_order(f, y0, t0, t0 + 4 * h, h)
    y_values[1:4] = y_rk[1:]

    f_values = np.zeros((len(t_values), len(y0)))
    f_values[0:4] = [f(y_values[i], t_values[i]) for i in range(4)]

    for i in range(4, len(t_values)):
        f_values[i - 1] = f(y_values[i - 1], t_values[i - 1])
        y_values[i] = y_values[i - 1] + h / 24 * (
            55 * f_values[i - 1]
            - 59 * f_values[i - 2]
            + 37 * f_values[i - 3]
            - 9 * f_values[i - 4]
        )

    return t_values, y_values


def adams_bashforth_moulton_4th_order(
    f: Callable[[list[float], list[float]], list[float]],
    y0: float,
    t0: float,
    t1: float,
    h: float,
):
    """Метод Адамса-Бешфорта-Моултона"""

    t_values = np.arange(t0, t1, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0

    t_rk, y_rk = runge_kutta_4th_order(f, y0, t0, t0 + 4 * h, h)
    y_values[1:4] = y_rk[1:]

    f_values = np.zeros((len(t_values), len(y0)))
    f_values[0:4] = [f(y_values[i], t_values[i]) for i in range(4)]

    for i in range(4, len(t_values)):
        f_values[i - 1] = f(y_values[i - 1], t_values[i - 1])
        pred = y_values[i - 1] + h / 24 * (
            55 * f_values[i - 1]
            - 59 * f_values[i - 2]
            + 37 * f_values[i - 3]
            - 9 * f_values[i - 4]
        )

        y_values[i] = y_values[i - 1] + h / 24 * (
            9 * f(pred, t_values[i])
            + 19 * f_values[i - 1]
            - 5 * f_values[i - 2]
            + f_values[i - 3]
        )

    return t_values, y_values


def gear_4th_order(
    f: Callable[[list[float], list[float]], list[float]],
    y0: float,
    t0: float,
    t1: float,
    h: float,
):
    """Метод Гира 4 порядка"""

    t_values = np.arange(t0, t1, h)
    num_steps = int((t1 - t0) / h) + 1
    y_values = np.zeros((num_steps, len(y0)))
    y_values[0] = y0

    p = 0
    t_rk, y_rk = runge_kutta_4th_order(f, y0, t0, t0 + 4 * h, h / 2**p)
    y_values[1:4] = (y_rk[:: 2**p])[1:4]

    for i in range(4, len(t_values)):

        error = [1, 1]
        y_values_iter = (
            48 * y_values[i - 1]
            - 36 * y_values[i - 2]
            + 16 * y_values[i - 3]
            - 3 * y_values[i - 4]
            + 12 * h * f(y_values[i - 1], t_values[i - 1])
        ) / 25
        y_values_iter_prev = y_values_iter + 1
        # print ("")
        while all(error[i] > 1e-13 for i in range(len(error))):

            y_values_iter = (
                48 * y_values[i - 1]
                - 36 * y_values[i - 2]
                + 16 * y_values[i - 3]
                - 3 * y_values[i - 4]
                + 12 * h * f(y_values_iter, 1)
            ) / 25
            error = np.abs((y_values_iter - y_values_iter_prev) / y_values_iter_prev)
            # print ("error:",error,"  cur:",y_values_iter,"  prev:",y_values_iter_prev)
            y_values_iter_prev = np.copy(y_values_iter)

        y_values[i] = (
            48 * y_values[i - 1]
            - 36 * y_values[i - 2]
            + 16 * y_values[i - 3]
            - 3 * y_values[i - 4]
            + 12 * h * f(y_values_iter, 1)
        ) / 25

    return t_values, y_values
