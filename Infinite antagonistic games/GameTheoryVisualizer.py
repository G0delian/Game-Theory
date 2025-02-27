"""
Date of code creation: 2025-21-02
Author: Uchkunov Boburmirzo
Modified by: Polat Tadjimurodov

Last modified: 2025-27-02
Last modified by: Uchkunov Boburmirzo

Description: Models a pursuit-evasion game between two players (P and E) acting optimally.
             Analyzes optimal strategies for various geometric shapes, from triangles to n-sided polygons.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, distance
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import random

# NEW: Для диалогового окна
import tkinter as tk
from tkinter import simpledialog

##############################
# 1) Алгоритм Уэлзла (Welzl) #
##############################
def ball_from_one_point(p, dim):
    return np.array(p, dtype=float), 0.0

def ball_from_two_points(p, q, dim):
    center = (np.array(p) + np.array(q)) / 2.0
    radius = distance.euclidean(p, q) / 2.0
    return center, radius

def ball_from_three_points_2d(a, b, c):
    A = np.array(a, dtype=float)
    B = np.array(b, dtype=float)
    C = np.array(c, dtype=float)

    d = 2.0 * (A[0] * (B[1] - C[1]) +
               B[0] * (C[1] - A[1]) +
               C[0] * (A[1] - B[1]))
    if abs(d) < 1e-14:
        # Точки почти на одной прямой, fallback
        return ball_from_two_points(a, b, 2)

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
    Упрощённый вариант для 3D: Берём окружность в проекции (x,y), z = среднее
    """
    center2d, radius2d = ball_from_three_points_2d(
        [a[0], a[1]], [b[0], b[1]], [c[0], c[1]]
    )
    z_mean = (a[2] + b[2] + c[2]) / 3.0
    center3d = np.array([center2d[0], center2d[1], z_mean], dtype=float)
    return center3d, radius2d

def ball_from_four_points_3d(a, b, c, d):
    """
    Простейший перебор четырёх точек: Смотрим все тройки, ищем наименьший радиус,
    в котором лежит и четвёртая точка. Иначе fallback.
    """
    A = np.array([a, b, c, d], dtype=float)
    combos = list(combinations(A, 3))
    best_center = None
    best_radius = float('inf')
    for combo in combos:
        c_tmp, r_tmp = ball_from_three_points_3d(*combo)
        all_inside = True
        for pt in A:
            if distance.euclidean(pt, c_tmp) > r_tmp + 1e-12:
                all_inside = False
                break
        if all_inside and r_tmp < best_radius:
            best_radius = r_tmp
            best_center = c_tmp

    if best_center is not None:
        return best_center, best_radius

    # fallback: если 4 точки слишком "в линию", используем ball_from_two_points
    max_dist = 0
    pair = (A[0], A[1])
    for i in range(4):
        for j in range(i+1, 4):
            dist_ij = distance.euclidean(A[i], A[j])
            if dist_ij > max_dist:
                max_dist = dist_ij
                pair = (A[i], A[j])
    return ball_from_two_points(pair[0], pair[1], 3)

def is_in_ball(p, center, radius):
    return distance.euclidean(p, center) <= radius + 1e-14

def ball_from_boundary(boundary, dim):
    if not boundary:
        return np.zeros(dim), 0.0
    elif len(boundary) == 1:
        return ball_from_one_point(boundary[0], dim)
    elif len(boundary) == 2:
        return ball_from_two_points(boundary[0], boundary[1], dim)
    elif dim == 2 and len(boundary) == 3:
        return ball_from_three_points_2d(boundary[0], boundary[1], boundary[2])
    elif dim == 3:
        if len(boundary) == 3:
            return ball_from_three_points_3d(boundary[0], boundary[1], boundary[2])
        if len(boundary) == 4:
            return ball_from_four_points_3d(boundary[0], boundary[1], boundary[2], boundary[3])
    return ball_from_one_point(boundary[0], dim)

def welzl(points, boundary, dim):
    """
    Рекурсивная функция Уэлзла для нахождения минимума опис. сферы.
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

##############################
# 2) Точка Торричелли (P)   #
##############################
def weiszfeld_geometric_median(points, max_iter=1000, eps=1e-7):
    """
    Метод Вейсфельда для нахождения точки (x, y),
    минимизирующей сумму расстояний до набора 2D-точек.
    """
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
            w = 1.0 / dist_
            numerator += p * w
            denominator += w
        new_pt = numerator / denominator
        if distance.euclidean(current, new_pt) < eps:
            return new_pt
        current = new_pt
    return current

class GameTheoryVisualizer:
    def __init__(self, dim=2):
        self.dim = dim
        self.points = []

        # Результаты
        self.center_O = None  # Центр окружности/сферы
        self.radius = 0.0
        self.center_P = None  # Точка Торричелли
        self.boundary_points = []  # Точки на границе -> стратег. E?

        self.fig = plt.figure(figsize=(10,8))
        if self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        # Объекты для графики
        self.points_scat = None
        self.sphere_plot = None
        self.O_text = None
        self.P_text = None
        self.E_texts = []
        self.convex_hull_plot = None
        self.hull_edge_texts = []

        self.info_text = None

        # Для управления мышью
        self.last_mouse_xdata = None
        self.last_mouse_ydata = None

        # Создадим второе окно для текстовой информации
        self.info_figure = None
        self.info_ax = None

        self.setup_plot()
        self.connect_events()

    def setup_plot(self):
        t = "2D Game" if self.dim == 2 else "3D Game"
        self.ax.set_title(f"{t}\n[P] добавить точку мышью | [M] добавить точку вручную | [C] пересчитать (O,P)")
        self.ax.set_xlim(-5,5)
        self.ax.set_ylim(-5,5)
        if self.dim == 3:
            self.ax.set_zlim(-5,5)

        # Scatter
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
        # 1) Добавление точки кликом мыши
        if event.key == 'p':
            if self.dim == 3:
                # NEW: Предупреждение
                print("В 3D режиме точку мышью ставить некорректно (z=0). Используйте [M] для ручного ввода.")
                return

            if self.last_mouse_xdata is not None and self.last_mouse_ydata is not None:
                if self.dim == 2:
                    pt = [self.last_mouse_xdata, self.last_mouse_ydata]
                else:
                    # В 3D тут было pt = [x, y, 0], что некорректно,
                    # поэтому убрали и просим ввод вручную
                    pass
                self.points.append(pt)
                print(f"Добавлена точка (2D) {pt}")
                self.update_plot()
            else:
                print("Курсор вне Axes, точку поставить нельзя")

        # 2) Добавление точки вручную через диалог
        elif event.key == 'm':
            self.add_point_manual()

        # 3) Пересчёт (O, P)
        elif event.key == 'c':
            if len(self.points) < 2:
                print("Недостаточно точек для расчёта!")
                return
            self.calculate_strategies()
            self.update_plot()
            self.show_info_window()

    # NEW: функция ручного добавления точки
    def add_point_manual(self):
        # Используем Tkinter для ввода координат
        root = tk.Tk()
        root.withdraw()

        if self.dim == 2:
            x = simpledialog.askfloat("Добавить точку", "Введите координату X:", parent=root)
            y = simpledialog.askfloat("Добавить точку", "Введите координату Y:", parent=root)
            if x is not None and y is not None:
                pt = [x, y]
                self.points.append(pt)
                print(f"Добавлена точка (2D) {pt}")
        else:
            x = simpledialog.askfloat("Добавить точку", "Введите координату X:", parent=root)
            y = simpledialog.askfloat("Добавить точку", "Введите координату Y:", parent=root)
            z = simpledialog.askfloat("Добавить точку", "Введите координату Z:", parent=root)
            if x is not None and y is not None and z is not None:
                pt = [x, y, z]
                self.points.append(pt)
                print(f"Добавлена точка (3D) {pt}")

        root.destroy()
        self.update_plot()

    def polygon_order_through_all_points(self, points):
        """
        Сортируем точки по полярному углу вокруг центроида — 
        чтобы построить "многоугольник" по ним (2D).
        """
        if len(points) < 3:
            return points.copy()
        arr = np.array(points)
        center_ = np.mean(arr, axis=0)
        angles = np.arctan2(arr[:,1]-center_[1], arr[:,0]-center_[0])
        inds = np.argsort(angles)
        return [points[i] for i in inds]

    def calculate_strategies(self):
        # Минимальная окружность/сфера
        self.center_O, self.radius = minimum_enclosing_sphere(self.points, self.dim)

        # Точка Торричелли
        if self.dim == 2:
            self.center_P = weiszfeld_geometric_median(self.points)
        else:
            # Для 3D берём xy и затем добавляем средний z
            xy = [[p[0], p[1]] for p in self.points]
            t_2d = weiszfeld_geometric_median(xy)
            mean_z = np.mean([p[2] for p in self.points])
            self.center_P = np.array([t_2d[0], t_2d[1], mean_z])

        # Точки на границе сферы -> E
        self.boundary_points = []
        for p in self.points:
            dist_ = distance.euclidean(p, self.center_O)
            if abs(dist_ - self.radius) <= 1e-5:
                self.boundary_points.append(p)

        # Для наглядности построим "многоугольник" (2D)
        if self.dim == 2 and len(self.points) >= 3:
            ordered_pts = self.polygon_order_through_all_points(self.points)
            closed = ordered_pts + [ordered_pts[0]]

            # Удаляем подписи рёбер
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

    def update_plot(self):
        # Обновляем scatter со всеми точками
        if self.dim == 2:
            self.points_scat.set_offsets(self.points)
        else:
            if self.points:
                x,y,z = zip(*self.points)
                self.points_scat._offsets3d = (x,y,z)
            else:
                self.points_scat._offsets3d = ([],[],[])

        # Удалим прежние объекты (окружность/сферу, тексты)
        if self.sphere_plot:
            try:
                self.sphere_plot.remove()  # 2D line или 3D surface
            except:
                pass
            self.sphere_plot = None
        if self.O_text:
            self.O_text.remove()
            self.O_text = None
        if self.P_text:
            self.P_text.remove()
            self.P_text = None
        for txt in self.E_texts:
            txt.remove()
        self.E_texts.clear()

        # 1) Рисуем окружность (2D) / сферу (3D)
        if self.radius>1e-9 and self.center_O is not None:
            if self.dim == 2:
                theta = np.linspace(0,2*np.pi,200)
                x_circ = self.center_O[0] + self.radius*np.cos(theta)
                y_circ = self.center_O[1] + self.radius*np.sin(theta)
                self.sphere_plot, = self.ax.plot(x_circ, y_circ, 'r-', lw=2, alpha=0.5)
            else:
                u = np.linspace(0,2*np.pi,60)
                v = np.linspace(0, np.pi,60)
                x_sph = self.center_O[0] + self.radius*np.outer(np.cos(u), np.sin(v))
                y_sph = self.center_O[1] + self.radius*np.outer(np.sin(u), np.sin(v))
                z_sph = self.center_O[2] + self.radius*np.outer(np.ones_like(u), np.cos(v))
                self.sphere_plot = self.ax.plot_surface(
                    x_sph, y_sph, z_sph, color='r', alpha=0.2
                )

        # 2) Подпись O
        if self.center_O is not None:
            if self.dim==2:
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

        # 3) Подпись P (Торричелли)
        if self.center_P is not None:
            if self.dim==2:
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

        # 4) Точки E на границе
        for pt in self.boundary_points:
            if self.dim==2:
                txt = self.ax.text(pt[0], pt[1], "E",
                                   color='black', fontsize=12, fontweight='bold',
                                   ha='center', va='center')
            else:
                txt = self.ax.text(pt[0], pt[1], pt[2], "E",
                                   color='black', fontsize=12, fontweight='bold',
                                   ha='center', va='center')
            self.E_texts.append(txt)

        # 5) Текст в углу
        info = (
            f"O = {self.center_O},  R={self.radius:.3f}\n"
            f"P = {self.center_P}\n"
            f"E: {len(self.boundary_points)} точек на границе\n"
        )
        self.info_text.set_text(info)

        self.fig.canvas.draw_idle()

    def show_info_window(self):
        """
        Открывает второе окошко, где текстом всё написано:
        - O, R, P, список E.
        """
        if self.info_figure is not None:
            plt.close(self.info_figure)

        self.info_figure = plt.figure(figsize=(4,3))
        self.info_ax = self.info_figure.add_subplot(111)
        self.info_ax.set_axis_off()

        lines = []
        lines.append(f"Центр минимального шара O = {self.center_O}")
        lines.append(f"Радиус = {self.radius:.3f}")
        lines.append(f"Точка Торричелли P = {self.center_P}")
        lines.append(f"Точки на границе (E):")
        if not self.boundary_points:
            lines.append("  - Нет точек на границе.")
        else:
            for i,pt in enumerate(self.boundary_points):
                lines.append(f"  E{i+1}: {pt}")

        lines.append("\nОптимальная стратегия E:")
        lines.append(" Если E стремится максимально удалиться от P, ")
        lines.append(" ему выгодно выбрать позицию (E) на границе шара.")

        full = "\n".join(lines)
        self.info_ax.text(0.05, 0.95, full, va='top', ha='left',
                          fontsize=10)

        self.info_ax.set_title("Информация об O и P")
        self.info_figure.tight_layout()
        self.info_figure.show()

def main():
    dim = 0
    while dim not in (2,3):
        try:
            dim = int(input("Выберите размерность (2 или 3): "))
        except:
            continue

    vis = GameTheoryVisualizer(dim=dim)
    plt.show()

if __name__=="__main__":
    main()
