"""
Date of code creation: 2025-21-02
Author:                Uchkunov Boburmirzo
Modified by:           Polat Tadjimurodov
Modified data:         2025-23-02

Last modified:         2025-27-02
Last modified by:      Uchkunov Boburmirzo

Description: Models a pursuit-evasion game between two players (P and E) acting optimally.
             Analyzes optimal strategies for various geometric shapes, from triangles to n-sided polygons.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from itertools import combinations
import random

import tkinter as tk
from tkinter import simpledialog

def ball_from_one_point(p):
    """
    Возвращает сферу радиуса 0, центр в точке p.
    """
    return np.array(p, dtype=float), 0.0

def ball_from_two_points(p, q):
    """
    Описывающая окружность/сфера для двух точек — центр посередине,
    радиус = половина расстояния.
    """
    center = (np.array(p) + np.array(q)) / 2.0
    radius = distance.euclidean(p, q) / 2.0
    return center, radius

def ball_from_three_points_2d(a, b, c):
    """
    Окружность через 3 точки в 2D.
    Если точки коллинеарны (почти), то fallback к ball_from_two_points.
    """
    A = np.array(a, dtype=float)
    B = np.array(b, dtype=float)
    C = np.array(c, dtype=float)

    d = 2.0 * (A[0] * (B[1] - C[1]) +
               B[0] * (C[1] - A[1]) +
               C[0] * (A[1] - B[1]))
    if abs(d) < 1e-14:
        return ball_from_two_points(a, b)

    ux = ((A[0]**2 + A[1]**2) * (B[1] - C[1]) +
          (B[0]**2 + B[1]**2) * (C[1] - A[1]) +
          (C[0]**2 + C[1]**2) * (A[1] - B[1])) / d
    uy = ((A[0]**2 + A[1]**2) * (C[0] - B[0]) +
          (B[0]**2 + B[1]**2) * (A[0] - C[0]) +
          (C[0]**2 + C[1]**2) * (B[0] - A[0])) / d
    center = np.array([ux, uy])
    radius = distance.euclidean(center, A)
    return center, radius

def ball_from_three_points_3d(a, b, c):
    """
    Сфера, проходящая через 3 точки в 3D.
    Реализован перевод в локальную плоскость и вычисление окружности в 2D.
    """
    A = np.array(a, dtype=float)
    B = np.array(b, dtype=float)
    C = np.array(c, dtype=float)

    AB = B - A
    AC = C - A

    normal = np.cross(AB, AC)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-14:
        # fallback к двум точкам
        return ball_from_two_points(a, b)

    normal /= norm_len

    ex = AB / np.linalg.norm(AB)
    ey = np.cross(normal, ex)

    def to_local(pt):
        v = pt - A
        return np.array([np.dot(v, ex), np.dot(v, ey)])

    A2d = np.array([0.0, 0.0])
    B2d = to_local(B)
    C2d = to_local(C)

    center2d, radius2d = ball_from_three_points_2d(A2d, B2d, C2d)
    center3d = A + center2d[0]*ex + center2d[1]*ey
    return center3d, radius2d

def ball_from_four_points_3d(a, b, c, d):
    """
    Сфера, проходящая через 4 точки в 3D.
    Если система вырождена, уходим в fallback.
    """
    A = np.array(a, dtype=float)
    B = np.array(b, dtype=float)
    C = np.array(c, dtype=float)
    D = np.array(d, dtype=float)

    def eq_sphere(p1, p2):
        return (
            p1[0]**2 + p1[1]**2 + p1[2]**2 - (p2[0]**2 + p2[1]**2 + p2[2]**2),
            -2*(p1[0]-p2[0]),
            -2*(p1[1]-p2[1]),
            -2*(p1[2]-p2[2])
        )

    eqAB = eq_sphere(A, B)
    eqAC = eq_sphere(A, C)
    eqAD = eq_sphere(A, D)

    M = [
        [eqAB[1], eqAB[2], eqAB[3]],
        [eqAC[1], eqAC[2], eqAC[3]],
        [eqAD[1], eqAD[2], eqAD[3]],
    ]
    v = [-eqAB[0], -eqAC[0], -eqAD[0]]
    M = np.array(M, dtype=float)
    v = np.array(v, dtype=float)

    try:
        sol = np.linalg.solve(M, v)
    except np.linalg.LinAlgError:
        return fallback_3d([A,B,C,D])

    center = sol
    r_2 = np.sum((center - A)**2)
    if r_2 < 0:
        r_2 = 0
    radius = np.sqrt(r_2)

    # Проверяем, лежат ли 4 точки на сфере
    for pt in [A,B,C,D]:
        dist_ = distance.euclidean(pt, center)
        if abs(dist_ - radius) > 1e-5:
            return fallback_3d([A,B,C,D])

    return center, radius

def fallback_3d(points):
    """
    Если 4 точки вырождены, пробуем все тройки и т.д.
    """
    arr = np.array(points)
    n = len(arr)
    best_center = None
    best_radius = float('inf')
    combos3 = list(combinations(range(n), 3))
    for c3 in combos3:
        tri = arr[list(c3)]
        center3d, radius3d = ball_from_three_points_3d(*tri)
        all_inside = True
        for i in range(n):
            dist_ = distance.euclidean(arr[i], center3d)
            if dist_ > radius3d + 1e-12:
                all_inside = False
                break
        if all_inside and radius3d < best_radius:
            best_radius = radius3d
            best_center = center3d

    if best_center is not None:
        return best_center, best_radius

    # Иначе две самые удалённые
    max_dist = 0
    pair = (arr[0], arr[0])
    for i in range(n):
        for j in range(i+1, n):
            dist_ij = distance.euclidean(arr[i], arr[j])
            if dist_ij > max_dist:
                max_dist = dist_ij
                pair = (arr[i], arr[j])
    return ball_from_two_points(pair[0], pair[1])

def is_in_ball(p, center, radius):
    return distance.euclidean(p, center) <= radius + 1e-14

def ball_from_boundary(boundary, dim):
    boundary = list(boundary)
    if len(boundary) == 0:
        if dim == 2:
            return np.zeros(2), 0.0
        else:
            return np.zeros(3), 0.0
    if len(boundary) == 1:
        return ball_from_one_point(boundary[0])
    if len(boundary) == 2:
        return ball_from_two_points(boundary[0], boundary[1])
    if dim == 2:
        return ball_from_three_points_2d(boundary[0], boundary[1], boundary[2])
    else:
        if len(boundary) == 3:
            return ball_from_three_points_3d(boundary[0], boundary[1], boundary[2])
        if len(boundary) == 4:
            return ball_from_four_points_3d(boundary[0], boundary[1], boundary[2], boundary[3])

    return ball_from_one_point(boundary[0])

def welzl(points, boundary, dim):
    """
    Рекурсивная функция Уэлзла для нахождения минимальной сферы
    """
    if not points or (dim == 2 and len(boundary) == 3) or (dim == 3 and len(boundary) == 4):
        return ball_from_boundary(boundary, dim)

    p = points.pop()
    center, radius = welzl(points, boundary, dim)
    if is_in_ball(p, center, radius):
        points.append(p)
        return center, radius

    boundary.append(p)
    center, radius = welzl(points, boundary, dim)
    boundary.pop()
    points.append(p)
    return center, radius

def minimum_enclosing_sphere(all_points, dim):
    pts_copy = all_points[:]
    random.shuffle(pts_copy)
    return welzl(pts_copy, [], dim)

def weiszfeld_geometric_median_2d(points, max_iter=1000, eps=1e-7):
    pts = np.array(points, dtype=float)
    if len(pts) == 0:
        return np.array([0.0, 0.0])
    current = np.mean(pts, axis=0)
    for _ in range(max_iter):
        numerator = np.zeros(2)
        denominator = 0.0
        for p in pts:
            dist_ = distance.euclidean(current, p)
            if dist_ < eps:
                return p
            w = 1.0 / (dist_ + 1e-14)
            numerator += p * w
            denominator += w
        new_pt = numerator / denominator
        if distance.euclidean(current, new_pt) < eps:
            return new_pt
        current = new_pt
    return current

def weiszfeld_geometric_median_3d(points, max_iter=1000, eps=1e-7):
    pts = np.array(points, dtype=float)
    if len(pts) == 0:
        return np.array([0.0, 0.0, 0.0])
    current = np.mean(pts, axis=0)
    for _ in range(max_iter):
        numerator = np.zeros(3)
        denominator = 0.0
        for p in pts:
            dist_ = distance.euclidean(current, p)
            if dist_ < eps:
                return p
            w = 1.0 / (dist_ + 1e-14)
            numerator += p * w
            denominator += w
        new_pt = numerator / denominator
        if distance.euclidean(current, new_pt) < eps:
            return new_pt
        current = new_pt
    return current

class GameTheoryVisualizer:
    """
    Класс для интерактивного добавления точек и визуализации:
    - Минимальной описывающей сферы (центр O, радиус R)
    - Геометрической медианы (точка P)
    - Точек E на границе
    """
    def __init__(self, dim=2):
        self.dim = dim
        self.points = []

        self.center_O = None
        self.radius = 0.0
        self.center_P = None
        self.boundary_points = []

        self.fig = plt.figure(figsize=(10,8))
        if self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        self.points_scat = None
        self.sphere_plot = None
        self.O_text = None
        self.P_text = None
        self.E_texts = []
        self.convex_hull_plot = None
        self.hull_edge_texts = []
        self.info_text = None

        self.last_mouse_xdata = None
        self.last_mouse_ydata = None

        self.info_figure = None
        self.info_ax = None

        self.setup_plot()
        self.connect_events()

    def setup_plot(self):
        t = "2D Game" if self.dim == 2 else "3D Game"
        self.ax.set_title(f"{t}\n[P] мышь (2D) | [M] ручной ввод | [C] пересчитать")
        self.ax.set_xlim(-5,5)
        self.ax.set_ylim(-5,5)
        if self.dim == 3:
            self.ax.set_zlim(-5,5)

        if self.dim == 2:
            self.points_scat = self.ax.scatter([], [], c='b', marker='o', s=50, label='Points')
            self.convex_hull_plot, = self.ax.plot([], [], 'm--', lw=1.5, alpha=0.7, label='Polygon')
            self.info_text = self.ax.text(
                0.03, 0.75, "",
                transform=self.ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=9
            )
        else:
            self.points_scat = self.ax.scatter([], [], c='b', marker='o', s=50)
            self.convex_hull_plot = None
            self.info_text = self.ax.text2D(
                0.03, 0.75, "",
                transform=self.ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=9
            )
        self.ax.legend(loc='upper right')

    def connect_events(self):
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            self.last_mouse_xdata = event.xdata
            self.last_mouse_ydata = event.ydata
        else:
            self.last_mouse_xdata = None
            self.last_mouse_ydata = None

    def on_key_press(self, event):
        if event.key == 'p':
            # Добавление точки по мыши (только 2D)
            if self.dim == 3:
                print("В 3D точку по мыши ставить нельзя. Используйте [M].")
                return
            if self.last_mouse_xdata is not None and self.last_mouse_ydata is not None:
                pt = [self.last_mouse_xdata, self.last_mouse_ydata]
                self.points.append(pt)
                print(f"Добавлена точка (2D) {pt}")
                self.update_plot()
            else:
                print("Курсор вне Axes. Точку поставить нельзя.")
        elif event.key == 'm':
            self.add_point_manual()
        elif event.key == 'c':
            if len(self.points) < 2:
                print("Недостаточно точек для расчёта!")
                return
            self.calculate_strategies()
            self.update_plot()
            self.show_info_window()

    def add_point_manual(self):
        root = tk.Tk()
        root.withdraw()

        try:
            if self.dim == 2:
                x = simpledialog.askfloat("Добавить точку", "Введите X:", parent=root)
                y = simpledialog.askfloat("Добавить точку", "Введите Y:", parent=root)
                if x is not None and y is not None:
                    pt = [x, y]
                    self.points.append(pt)
                    print(f"Добавлена точка (2D) {pt}")
            else:
                x = simpledialog.askfloat("Добавить точку", "Введите X:", parent=root)
                y = simpledialog.askfloat("Добавить точку", "Введите Y:", parent=root)
                z = simpledialog.askfloat("Добавить точку", "Введите Z:", parent=root)
                if x is not None and y is not None and z is not None:
                    pt = [x, y, z]
                    self.points.append(pt)
                    print(f"Добавлена точка (3D) {pt}")
        except:
            print("Ошибка при вводе координат.")
        finally:
            root.destroy()

        # Если нужно, чтобы сразу отображалась точка:
        self.update_plot()

    def polygon_order_through_all_points(self, points):
        if len(points) < 3:
            return points.copy()
        arr = np.array(points)
        center_ = np.mean(arr, axis=0)
        angles = np.arctan2(arr[:,1]-center_[1], arr[:,0]-center_[0])
        inds = np.argsort(angles)
        return [points[i] for i in inds]

    def calculate_strategies(self):
        self.center_O, self.radius = minimum_enclosing_sphere(self.points, self.dim)

        if self.dim == 2:
            self.center_P = weiszfeld_geometric_median_2d(self.points)
        else:
            self.center_P = weiszfeld_geometric_median_3d(self.points)

        self.boundary_points = []
        for p in self.points:
            dist_ = distance.euclidean(p, self.center_O)
            if abs(dist_ - self.radius) <= 1e-5:
                self.boundary_points.append(p)

        if self.dim == 2 and len(self.points) >= 3:
            ordered_pts = self.polygon_order_through_all_points(self.points)
            closed = ordered_pts + [ordered_pts[0]]

            for txt in self.hull_edge_texts:
                txt.remove()
            self.hull_edge_texts.clear()

            xh,yh = zip(*closed)
            self.convex_hull_plot.set_data(xh, yh)

            for i in range(len(ordered_pts)):
                p1 = closed[i]
                p2 = closed[i+1]
                l_ = distance.euclidean(p1,p2)
                mx = 0.5*(p1[0]+p2[0])
                my = 0.5*(p1[1]+p2[1])
                txt = self.ax.text(mx, my, f"{l_:.2f}", color='m', fontsize=9,
                                   ha='center', va='center')
                self.hull_edge_texts.append(txt)
        else:
            if self.convex_hull_plot is not None:
                self.convex_hull_plot.set_data([], [])

    def update_plot(self):
        # Обновляем точки
        if self.dim == 2:
            self.points_scat.set_offsets(self.points)
        else:
            if len(self.points) > 0:
                x, y, z = zip(*self.points)
                self.points_scat._offsets3d = (x,y,z)
            else:
                self.points_scat._offsets3d = ([],[],[])

        # Убираем предыдущую сферу (если была)
        if self.sphere_plot:
            try:
                self.sphere_plot.remove()
            except:
                pass
            self.sphere_plot = None

        # Убираем предыдущие подписи (O, P, E)
        if self.O_text:
            self.O_text.remove()
            self.O_text = None
        if self.P_text:
            self.P_text.remove()
            self.P_text = None
        for txt in self.E_texts:
            txt.remove()
        self.E_texts.clear()

        # Если есть центр и радиус, рисуем окружность/сферу
        if self.center_O is not None and self.radius > 1e-12:
            if self.dim == 2:
                theta = np.linspace(0, 2*np.pi, 200)
                x_circ = self.center_O[0] + self.radius*np.cos(theta)
                y_circ = self.center_O[1] + self.radius*np.sin(theta)
                self.sphere_plot, = self.ax.plot(x_circ, y_circ, 'r-', lw=2, alpha=0.5)
            else:
                # Рисуем сферу в 3D: «проволочный каркас» (wireframe)
                u = np.linspace(0, 2*np.pi, 60)
                v = np.linspace(0, np.pi, 60)
                x_sph = self.center_O[0] + self.radius*np.outer(np.cos(u), np.sin(v))
                y_sph = self.center_O[1] + self.radius*np.outer(np.sin(u), np.sin(v))
                z_sph = self.center_O[2] + self.radius*np.outer(np.ones_like(u), np.cos(v))

                # Добавим каркас (wireframe). Можно добавить и plot_surface отдельно, если нужно.
                self.sphere_plot = self.ax.plot_wireframe(
                    x_sph, y_sph, z_sph,
                    rstride=6,
                    cstride=6,
                    color='r',
                    alpha=0.6
                )

                # Дополнительно провести «спицы» — линии от центра O к точкам на границе E
                for pt in self.boundary_points:
                    self.ax.plot(
                        [self.center_O[0], pt[0]],
                        [self.center_O[1], pt[1]],
                        [self.center_O[2], pt[2]],
                        color='k', linestyle='--'
                    )

        # Подпись центра O
        if self.center_O is not None:
            if self.dim == 2:
                self.O_text = self.ax.text(
                    self.center_O[0], self.center_O[1], "O",
                    color='red', fontsize=12, fontweight='bold',
                    ha='center', va='center'
                )
            else:
                self.O_text = self.ax.text(
                    self.center_O[0], self.center_O[1], self.center_O[2],
                    "O", color='red', fontsize=12, fontweight='bold',
                    ha='center', va='center'
                )

        # Подпись геометрической медианы P
        if self.center_P is not None:
            if self.dim == 2:
                self.P_text = self.ax.text(
                    self.center_P[0], self.center_P[1], "P",
                    color='blue', fontsize=12, fontweight='bold',
                    ha='center', va='center'
                )
            else:
                self.P_text = self.ax.text(
                    self.center_P[0], self.center_P[1], self.center_P[2],
                    "P", color='blue', fontsize=12, fontweight='bold',
                    ha='center', va='center'
                )

        # Подписи точек на границе E
        for pt in self.boundary_points:
            if self.dim == 2:
                txt = self.ax.text(pt[0], pt[1], "E", color='black',
                                   fontsize=12, fontweight='bold',
                                   ha='center', va='center')
            else:
                txt = self.ax.text(pt[0], pt[1], pt[2],
                                   "E", color='black',
                                   fontsize=12, fontweight='bold',
                                   ha='center', va='center')
            self.E_texts.append(txt)

        # Информация об O, R, P, E
        center_o_str = (str(np.round(self.center_O,3)) 
                        if self.center_O is not None else "None")
        center_p_str = (str(np.round(self.center_P,3)) 
                        if self.center_P is not None else "None")

        info = (
            f"O = {center_o_str}, R={self.radius:.3f}\n"
            f"P = {center_p_str}\n"
            f"E: {len(self.boundary_points)} точек на границе\n"
        )
        self.info_text.set_text(info)

        self.fig.canvas.draw_idle()

    def show_info_window(self):
        if self.info_figure is not None:
            plt.close(self.info_figure)

        self.info_figure = plt.figure(figsize=(4,3))
        self.info_ax = self.info_figure.add_subplot(111)
        self.info_ax.set_axis_off()

        lines = []

        if self.center_O is not None:
            lines.append(f"Центр сферы O = {np.round(self.center_O,3)}")
        else:
            lines.append("Центр сферы O = None")
        lines.append(f"Радиус = {self.radius:.3f}")
        
        if self.center_P is not None:
            lines.append(f"Геом. медиана P = {np.round(self.center_P,3)}")
        else:
            lines.append("Геом. медиана P = None")

        lines.append(f"Точек на границе: {len(self.boundary_points)}")
        for i,pt in enumerate(self.boundary_points):
            lines.append(f"  E{i+1}: {np.round(pt,3)}")

        lines.append("\nОптимальная стратегия E — максимально удалиться от P на сфере.")

        full = "\n".join(lines)
        self.info_ax.text(0.05, 0.95, full, va='top', ha='left', fontsize=10)
        self.info_ax.set_title("Информация об O и P")

        self.info_figure.tight_layout()
        self.info_figure.show()

def main():
    dim = 0
    while dim not in (2, 3):
        try:
            dim = int(input("Выберите размерность (2 или 3): "))
        except:
            pass

    vis = GameTheoryVisualizer(dim=dim)
    plt.show()

if __name__ == "__main__":
    main()
