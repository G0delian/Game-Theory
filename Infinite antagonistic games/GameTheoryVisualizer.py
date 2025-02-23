"""
Date of code creation: 2024-21-02
Author: Uchkunov Boburmirzo 
Last modified: 2024-24-01
Modified by: Polat Tadjimurodov

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
    center2d, radius2d = ball_from_three_points_2d(
        [a[0], a[1]], [b[0], b[1]], [c[0], c[1]]
    )
    z_mean = (a[2] + b[2] + c[2]) / 3.0
    center3d = np.array([center2d[0], center2d[1], z_mean], dtype=float)
    return center3d, radius2d

def ball_from_four_points_3d(a, b, c, d):
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

    # fallback — если 4 точки почти копланарны
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

class GameTheoryVisualizer:
    def __init__(self, dim=2):
        self.dim = dim
        self.points = []
        self.radius = 0.0
        self.center = None
        self.boundary_points = []
        self.expected_distance = 0.0
        self.convex_hull_info = ""

        self.manual_lines = []
        self.selected_line_points = []

        self.hull_edge_texts = []
        self.last_mouse_xdata = None
        self.last_mouse_ydata = None

        self.E_texts = []
        self.center_text = None
        self.sphere_plot = None
        self.fig = plt.figure(figsize=(10, 8))

        if self.dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        self.setup_ui()

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def setup_ui(self):
        title = "2D Игра" if self.dim == 2 else "3D Игра"
        self.ax.set_title(
            f"{title}\n"
            "P: поставить точку у курсора | C: пересчитать стратегию"
        )
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        if self.dim == 3:
            self.ax.set_zlim(-5, 5)

        self.points_scat = self.ax.scatter([], [], c='b', marker='o', s=50,
                                           label='Все точки')
        if self.dim == 2:
            self.convex_hull_plot, = self.ax.plot([], [], 'm--', lw=1.5, alpha=0.7,
                                                  label='Многоугольник')
        else:
            self.convex_hull_plot = None

        if self.dim == 2:
            self.info_text = self.ax.text(
                0.03, 0.75, "",
                transform=self.ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=9
            )
        else:
            self.info_text = self.ax.text2D(
                0.03, 0.75, "",
                transform=self.ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=9
            )

        self.ax.legend(loc='upper right')

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            self.last_mouse_xdata = event.xdata
            self.last_mouse_ydata = event.ydata
        else:
            self.last_mouse_xdata = None
            self.last_mouse_ydata = None

    def on_key_press(self, event):
        if event.key == 'p':
            if self.last_mouse_xdata is not None and self.last_mouse_ydata is not None:
                if self.dim == 2:
                    new_point = [self.last_mouse_xdata, self.last_mouse_ydata]
                else:
                    new_point = [self.last_mouse_xdata, self.last_mouse_ydata, 0.0]
                self.points.append(new_point)
                print(f"Добавлена точка {new_point}")
                self.update_plot()
            else:
                print("Курсор вне области Axes, точку поставить нельзя.")

        elif event.key == 'c':
            if len(self.points) < 2:
                print("Недостаточно точек для расчёта!")
            else:
                self.calculate_strategies()
                self.update_plot()

    def add_manual_line(self, two_points):
        p1, p2 = two_points
        length = distance.euclidean(p1, p2)

        if self.dim == 2:
            line_obj, = self.ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                '--', color='orange', alpha=0.7
            )
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2
            text_obj = self.ax.text(
                mx, my, f"{length:.2f}",
                color='orange', fontsize=9,
                ha='center', va='center'
            )
        else:
            x_vals = [p1[0], p2[0]]
            y_vals = [p1[1], p2[1]]
            z_vals = [p1[2], p2[2]]
            line_obj, = self.ax.plot3D(
                x_vals, y_vals, z_vals,
                '--', color='orange', alpha=0.7
            )
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2
            mz = (p1[2] + p2[2]) / 2
            text_obj = self.ax.text(
                mx, my, mz, f"{length:.2f}",
                color='orange', fontsize=9,
                ha='center', va='center'
            )

        self.manual_lines.append({
            'object': line_obj,
            'length_text': text_obj
        })

    # ========================================================================
    # Новый метод для упорядочивания всех точек Let's try method
    # ========================================================================
    def polygon_order_through_all_points(self, points):
        """
        Упорядочивает список points по возрастанию угла относительно центроида.
        Возвращает отсортированный список.
        """
        if len(points) < 3:
            return points.copy()
        pts = np.array(points)
        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        order = np.argsort(angles)
        return [points[i] for i in order]
    # ========================================================================

    def calculate_strategies(self):
        all_pts_array = np.array(self.points)

        if self.dim == 2:
						# your method
            if len(self.points) >= 2:
                ordered_pts = self.polygon_order_through_all_points(self.points)
                ordered_pts_closed = ordered_pts + [ordered_pts[0]]

                for txt in self.hull_edge_texts:
                    txt.remove()
                self.hull_edge_texts.clear()

                xh, yh = zip(*ordered_pts_closed)
                self.convex_hull_plot.set_data(xh, yh)

                for i in range(len(ordered_pts)):
                    p1 = ordered_pts_closed[i]
                    p2 = ordered_pts_closed[i+1]
                    length = distance.euclidean(p1, p2)
                    mx = (p1[0] + p2[0]) / 2
                    my = (p1[1] + p2[1]) / 2
                    t = self.ax.text(mx, my, f"{length:.2f}",
                                     color='m', fontsize=9,
                                     ha='center', va='center')
                    self.hull_edge_texts.append(t)

                self.convex_hull_info = "Построен многоугольник по всем точкам."
            else:
                self.convex_hull_info = "Недостаточно точек для построения многоугольника."
        else:
            if len(self.points) >= 4:
                try:
                    hull = ConvexHull(all_pts_array)
                    self.convex_hull_info = "Выпуклая оболочка: расчёт выполнен."
                except:
                    self.convex_hull_info = "Не удалось построить выпуклую оболочку."
            else:
                self.convex_hull_info = "Недостаточно точек для выпуклой оболочки в 3D."

        self.center, self.radius = minimum_enclosing_sphere(self.points, self.dim)
        self.boundary_points = []
        for pt in self.points:
            dist_pt = distance.euclidean(pt, self.center)
            if abs(dist_pt - self.radius) <= 1e-5:
                self.boundary_points.append(pt)

        if self.boundary_points:
            self.expected_distance = np.mean([
                distance.euclidean(p, self.center) for p in self.boundary_points
            ])
        else:
            self.expected_distance = 0.0

    def update_plot(self):
        if self.dim == 2:
            self.points_scat.set_offsets(self.points)
        else:
            if self.points:
                x, y, z = zip(*self.points)
                self.points_scat._offsets3d = (x, y, z)
            else:
                self.points_scat._offsets3d = ([], [], [])

        for txt in self.E_texts:
            txt.remove()
        self.E_texts.clear()

        if self.center_text:
            self.center_text.remove()
            self.center_text = None

        # Помечаем граничные точки (E)
        for pt in self.boundary_points:
            if self.dim == 2:
                txt = self.ax.text(pt[0], pt[1], 'E', color='black',
                                   fontsize=12, fontweight='bold',
                                   ha='center', va='center')
            else:
                txt = self.ax.text(pt[0], pt[1], pt[2], 'E', color='black',
                                   fontsize=12, fontweight='bold',
                                   ha='center', va='center')
            self.E_texts.append(txt)

        # Центр (P)
        if self.center is not None:
            if self.dim == 2:
                self.center_text = self.ax.text(
                    self.center[0], self.center[1], 'P',
                    color='black', fontsize=12, fontweight='bold',
                    ha='center', va='center'
                )
            else:
                self.center_text = self.ax.text(
                    self.center[0], self.center[1], self.center[2], 'P',
                    color='black', fontsize=12, fontweight='bold',
                    ha='center', va='center'
                )

        if self.sphere_plot:
            self.sphere_plot.remove()
            self.sphere_plot = None

        if self.radius > 1e-9 and self.center is not None:
            if self.dim == 2:
                theta = np.linspace(0, 2*np.pi, 200)
                x_circle = self.center[0] + self.radius * np.cos(theta)
                y_circle = self.center[1] + self.radius * np.sin(theta)
                self.sphere_plot, = self.ax.plot(
                    x_circle, y_circle, 'r-', lw=2, alpha=0.5
                )
            else:
                u = np.linspace(0, 2*np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                x_sph = self.center[0] + self.radius * np.outer(np.cos(u), np.sin(v))
                y_sph = self.center[1] + self.radius * np.outer(np.sin(u), np.sin(v))
                z_sph = self.center[2] + self.radius * np.outer(
                    np.ones_like(u), np.cos(v)
                )
                self.sphere_plot = self.ax.plot_surface(
                    x_sph, y_sph, z_sph, color='r', alpha=0.2
                )

        info_text = (
            f"Радиус окружности/сферы: {self.radius:.2f}\n"
            f"Ожидаемое расстояние (среднее по граничным точкам): {self.expected_distance:.2f}\n"
            f"{self.convex_hull_info}"
        )
        self.info_text.set_text(info_text)

        self.fig.canvas.draw_idle()

def main():
    dim = 0
    while dim not in (2, 3):
        try:
            dim = int(input("Выберите размерность (2/3): "))
        except:
            continue

    visualizer = GameTheoryVisualizer(dim=dim)
    plt.show()

if __name__ == "__main__":
    main()
