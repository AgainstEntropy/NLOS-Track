import os
import time
# from collections import namedtuple

import numpy as np
from numpy import ndarray
from scipy.io import savemat, loadmat
from shapely.geometry import Polygon, Point

from utils.vis import draw_route


class route_generator(object):
    def __init__(self, map_size: tuple[float] = (5, 5), forbidden_rate: float = 0.1):
        self.forbidden_rate = forbidden_rate
        self.route_length = None
        self.map_size = map_size

        x_min, x_max, y_min, y_max = (map_size[0] * self.forbidden_rate,
                                      map_size[0] * (1 - self.forbidden_rate),
                                      map_size[1] * self.forbidden_rate,
                                      map_size[1] * (1 - self.forbidden_rate))
        self.boundary = Polygon(((x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)))

        self.e_position = None
        self.e_route = None
        self.c_route = None
        self.velocities = None

        # 0.035m pre frame
        self.v_range = (0.03, 0.04)

    def _init_pv(self):
        bounds = self.boundary.bounds
        x = bounds[0] + (bounds[2] - bounds[0]) * np.random.rand()
        y = bounds[1] + (bounds[3] - bounds[1]) * np.random.rand()
        self.e_position = np.array([x, y])
        self.e_route = [self.e_position.copy()]

        self.velocity = np.random.rand(2).astype(np.float32) - 0.5
        self.velocity = 0.035 * self.velocity / np.linalg.norm(self.velocity)
        self.velocities = [self.velocity.copy()]

    def generate_route(self,
                       route_length: int = 256,
                       turn_rate: float = 0.15,
                       verbose: bool = False):
        self.route_length = route_length

        self._init_pv()
        for step in range(route_length):
            # print(self.velocity)
            self.next_step(turn_rate=turn_rate)
            self.e_route.append(self.e_position.copy())
            self.velocities.append(self.velocity.copy())

        self.c_route = [(self.e_route[i] + self.e_route[i + 1]) / 2 for i in range(len(self.e_route) - 1)]

        if verbose:
            print(len(self.velocities), len(self.c_route))
            print('velocities\n', np.stack(self.velocities))
            print('route:\n', np.stack(self.c_route))

    def next_step(self, turn_rate: float):
        self.e_position += self.velocity
        self.check_boundary()

        delta_v = np.random.rand(2).astype(np.float32) - 0.5
        delta_v /= np.linalg.norm(delta_v)

        v_norm = self.v_range[0] + (self.v_range[1] - self.v_range[0]) * np.random.rand()
        self.velocity += turn_rate * v_norm * delta_v
        self.velocity *= v_norm / np.linalg.norm(self.velocity)

    def check_boundary(self):
        point = Point(self.e_position)
        if not self.boundary.contains(point):
            for i in range(2):
                p = self.e_position[i]
                bound = self.boundary.bounds[i::2]
                if p < min(bound) or p > max(bound):
                    self.velocity[i] *= -1
                    break
        self.e_position = self.e_route[-1] + self.velocity

    def draw_route(self, cmap: str = 'viridis', normalize: bool = True):
        route = np.stack(self.c_route) / np.array(self.map_size)
        map_size = np.array((1, 1)) if normalize else self.map_size
        draw_route(map_size, route, cmap, return_mode=None)

    def save_route(self, save_root: str, verbose: bool = False):

        time.sleep(1)
        save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_dir = os.path.join(save_root, save_time)
        os.makedirs(save_dir, exist_ok=True)
        mat_path = os.path.join(save_dir, 'route.mat')
        map_size = np.array(self.map_size)
        # print(map_size)
        save_dict = {"map_size": map_size,
                     "route": np.stack(self.c_route) / map_size,  # (T, 2)
                     "velocities": np.stack(self.velocities[:-1])}

        savemat(mat_path, save_dict)
        if verbose:
            print(f'Save data into {mat_path} successfully!')

    def load_route(self,
                   mat_name: str,
                   save_dir: str):
        if not os.path.exists(save_dir):
            print(f"{save_dir} doesn't exist!")
        if mat_name is None:
            mat_names = sorted([f for f in os.listdir(save_dir) if f.endswith('.mat')])
            mat_name = mat_names[-1]
        mat_path = os.path.join(save_dir, mat_name)
        save_dict = loadmat(mat_path)

        self.e_route = [p for p in save_dict['route']]
        self.velocities = [v for v in save_dict['velocities']]
        print(f'Load data from {mat_path} successfully!')


def fix_real_trajectory(route_clip: ndarray, threshold: int = 10):
    bad_frames = find_miss_points(route_clip)
    b_len = len(bad_frames)
    counter = 1
    for i, frame in enumerate(bad_frames):
        if frame != 0:
            if i < b_len - 1 and bad_frames[i + 1] == frame + 1:
                counter += 1
            else:
                if counter <= threshold:
                    div = counter + 1
                    start_frame, end_frame = frame - counter, frame + 1
                    for j in range(1, div):
                        idx = start_frame + j
                        route_clip[idx] = route_clip[start_frame] * (j / div) \
                                          + route_clip[end_frame] * (1 - j / div)
                counter = 1

    return route_clip


def find_miss_points(route_clip: ndarray):
    return np.argwhere(route_clip == 0)[::2, 0]
