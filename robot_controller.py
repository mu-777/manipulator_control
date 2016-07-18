#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lib.Leap as Leap

# units: mm, rad

LINK_LENGTHES_MM = [45, 45, 45, 45, 45]


def to_vector(x, y, z):
    return np.matrix([x, y, z]).T


class Robot:
    def __init__(self, init_angs_rad):
        self._joints_pos = [to_vector(0, 0, 0), to_vector(0, 0, 0),
                            to_vector(0, 0, 0), to_vector(0, 0, 0),
                            to_vector(0, 0, 0), to_vector(0, 0, 0)]
        self._angles_rad = init_angs_rad
        self.input_angles(init_angs_rad)

    def input_angles(self, input_angs_rad):
        self._angles_rad = input_angs_rad

        input_angs_rad = np.append(input_angs_rad, np.matrix([[0]]), axis=0)
        trans = np.matrix(np.identity(4))
        for idx, (t, ang) in enumerate(zip([self._t0to1, self._t1to2, self._t2to3, self._t3to4, self._t4toTip],
                                           input_angs_rad)):
            trans *= t(ang[0, 0])
            self._joints_pos[idx + 1] = np.copy(trans[0:3, 3:4])
        return self

    def get_angles_rad(self):
        return self._angles_rad

    @property
    def joint_pos(self):
        return self._joints_pos

    def _t0to1(self, ang):
        s, c, l, th = np.sin, np.cos, LINK_LENGTHES_MM, ang
        return np.matrix([[s(th), c(th), 0, 0], [0, 0, 1, 0], [c(th), -s(th), 0, l[0]], [0, 0, 0, 1]])

    def _t1to2(self, ang):
        s, c, l, th = np.sin, np.cos, LINK_LENGTHES_MM, ang
        return np.matrix([[c(th), -s(th), 0, l[1]], [0, 0, 1, 0], [-s(th), -c(th), 0, 0], [0, 0, 0, 1]])

    def _t2to3(self, ang):
        s, c, l, th = np.sin, np.cos, LINK_LENGTHES_MM, ang
        return np.matrix([[c(th), -s(th), 0, l[2]], [0, 0, -1, 0], [s(th), c(th), 0, 0], [0, 0, 0, 1]])

    def _t3to4(self, ang):
        s, c, l, th = np.sin, np.cos, LINK_LENGTHES_MM, ang
        return np.matrix([[c(th), -s(th), 0, l[3]], [0, 0, 1, 0], [-s(th), -c(th), 0, 0], [0, 0, 0, 1]])

    def _t4toTip(self, ang):
        s, c, l, th = np.sin, np.cos, LINK_LENGTHES_MM, ang
        return np.matrix([[1, 0, 0, l[4]], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


class RobotController:
    MAX_CALC_ITER = 1000

    class RobotModel:
        _length = LINK_LENGTHES_MM

        def __init__(self, init_angs_rad):
            self.angles_rad = init_angs_rad

        def get_ee_pos(self, angs_rad):
            def s(i):
                return np.sin(angs_rad[i - 1, 0])

            def c(i):
                return np.cos(angs_rad[i - 1, 0])

            def l(i):
                return self._length[i]

            x = l(1) * s(1) + l(2) * s(1) * c(2) + l(3) * (s(1) * c(2) * c(3) + c(1) * s(3)) + \
                l(4) * (s(1) * c(2) * c(3) * c(4) + c(1) * s(3) * c(4) - s(1) * s(2) * s(4))
            y = -l(2) * s(2) - l(3) * s(2) * c(3) - l(4) * (s(2) * c(3) * c(4) + c(2) * s(4))
            z = l(0) + l(1) * c(1) + l(2) * c(1) * c(2) + l(3) * (c(1) * c(2) * c(3) - s(1) * s(3)) + \
                l(4) * (c(1) * c(2) * c(3) * c(4) - s(1) * s(3) * c(4) - c(1) * s(2) * s(4))
            return to_vector(x, y, z)

        def get_jacobi(self, angs_rad):
            def s(i):
                return np.sin(angs_rad[i - 1, 0])

            def c(i):
                return np.cos(angs_rad[i - 1, 0])

            def l(i):
                return self._length[i]

            jacobi = np.matrix(np.zeros((3, 4)))

            jacobi[0, 0] = l(1) * c(1) + l(2) * c(1) * c(2) + l(3) * (c(1) * c(2) * c(3) - s(1) * s(3)) + \
                           l(4) * (c(1) * c(2) * c(3) * c(4) - s(1) * s(3) * c(4) - c(1) * s(2) * s(4))
            jacobi[0, 1] = -l(2) * s(1) * s(2) - l(3) * s(1) * s(2) * c(3) - \
                           l(4) * (s(1) * s(2) * c(3) * c(4) + s(1) * c(2) * s(4))
            jacobi[0, 2] = -l(3) * (s(1) * c(2) * s(3) - c(1) * c(3)) - \
                           l(4) * (s(1) * c(2) * s(3) * c(4) - c(1) * c(3) * c(4))
            jacobi[0, 3] = -l(4) * (s(1) * c(2) * c(3) * s(4) + c(1) * s(3) * s(4) + s(1) * s(2) * c(4))

            jacobi[1, 0] = 0.0
            jacobi[1, 1] = -l(2) * c(2) - l(3) * c(2) * c(3) - l(4) * (c(2) * c(3) * c(4) - s(2) * s(4))
            jacobi[1, 2] = l(3) * s(2) * s(3) + l(4) * s(2) * s(3) * c(4)
            jacobi[1, 3] = l(4) * (s(2) * c(3) * s(4) - c(2) * c(4))

            jacobi[2, 0] = -l(1) * s(1) - l(2) * s(1) * c(2) - l(3) * (s(1) * c(2) * c(3) + c(1) * s(3)) - \
                           l(4) * (s(1) * c(2) * c(3) * c(4) + c(1) * s(3) * c(4) - s(1) * s(2) * s(4))
            jacobi[2, 1] = -l(2) * c(1) * s(2) - l(3) * c(1) * s(2) * c(3) - \
                           l(4) * (c(1) * s(2) * c(3) * c(4) + c(1) * c(2) * s(4))
            jacobi[2, 2] = -l(3) * (c(1) * c(2) * s(3) + s(1) * c(3)) - \
                           l(4) * (c(1) * c(2) * s(3) * c(4) + s(1) * c(3) * c(4))
            jacobi[2, 3] = -l(4) * (c(1) * c(2) * c(3) * s(4) - s(1) * s(3) * s(4) + c(1) * s(2) * c(4))

            return jacobi

    def __init__(self, init_angs_rad):
        self._robot_model = self.RobotModel(init_angs_rad)
        self._input_angles_rad = init_angs_rad

    def update(self, curr_angles, ref_ee_pos):
        angles = curr_angles
        for i in xrange(self.MAX_CALC_ITER):
            try:
                jacobi = self._robot_model.get_jacobi(angles)
                inv_jacobi = jacobi.T * np.linalg.inv(jacobi * jacobi.T)
                angles = angles - inv_jacobi * (self._robot_model.get_ee_pos(angles) - ref_ee_pos)
            except np.linalg.LinAlgError:
                self._input_angles_rad = angles
                break
            if (self._is_close(angles, ref_ee_pos)):
                self._input_angles_rad = angles
                break
        else:
            self._input_angles_rad = curr_angles
        return self

    @property
    def input_angles_rad(self):
        return self._input_angles_rad

    def _is_close(self, angles, ref_ee_pos, epsilon=0.0001):
        cur_ee_pos = self._robot_model.get_ee_pos(angles)
        diff = np.linalg.norm(cur_ee_pos - ref_ee_pos)
        normalized = diff / np.linalg.norm(ref_ee_pos)
        return normalized < epsilon


class Visualizer:
    DRAW_INTERVAL_SEC = 0.1

    def __init__(self, fig_num=1):
        self._fig = plt.figure(fig_num)
        self._ax = Axes3D(self._fig)

    def _set_drawing_params(self):
        self._ax.set_xlim(left=-sum(LINK_LENGTHES_MM) / 2,
                          right=sum(LINK_LENGTHES_MM) / 2)
        self._ax.set_ylim(bottom=-sum(LINK_LENGTHES_MM) / 2,
                          top=sum(LINK_LENGTHES_MM) / 2)
        self._ax.set_zlim(bottom=0,
                          top=sum(LINK_LENGTHES_MM))

    def draw(self):
        self._set_drawing_params()
        plt.pause(self.DRAW_INTERVAL_SEC)
        self._ax.clear()

    def add_robot(self, joints_pos):
        x, y, z = [], [], []
        for jpos in joints_pos:
            x.append(jpos[0, 0])
            y.append(jpos[1, 0])
            z.append(jpos[2, 0])
        self._ax.plot3D(x, y, z, linewidth=5, c='b')
        self._ax.scatter(x, y, z, s=100, c='c')
        return self

    def add_point(self, x, y, z):
        self._ax.scatter(x, y, z, s=300, c='r')
        return self


class LeapManager:
    def __init__(self, offset_from_origin_mm=None):
        self._leap = Leap.Controller()

        offset = to_vector(0, 0, 0) if offset_from_origin_mm is None else offset_from_origin_mm
        self._trans = np.matrix([[1, 0, 0, offset[0, 0]],
                                 [0, np.cos(np.deg2rad(90)), -np.sin(np.deg2rad(90)), offset[1, 0]],
                                 [0, np.sin(np.deg2rad(90)), np.cos(np.deg2rad(90)), offset[2, 0]],
                                 [0, 0, 0, 1]])

        self._palms = []

    def update(self):
        f = self._leap.frame()

        if len(f.hands) == 0:
            return False

        self._palms = [to_vector(hand.palm_position.x,
                                 hand.palm_position.y,
                                 hand.palm_position.z) for hand in f.hands]
        return True

    @property
    def palms(self):
        return [self._transform(self._trans, palm) for palm in self._palms]

    @staticmethod
    def _transform(trans, vec):
        return (trans * np.append(vec, np.matrix([[1]]), axis=0))[0:3, 0:1]


# --------------------------------------------
if __name__ == '__main__':
    init_angs_rad = np.matrix(np.deg2rad([25, 15, -35, 60])).T

    robot_system = Robot(init_angs_rad)

    leap_mgr = LeapManager(np.matrix([0, -100, 0]).T)
    robot_controller = RobotController(robot_system.get_angles_rad())
    robot_visualizer = Visualizer()

    target_pos = robot_system.joint_pos[-1]

    while True:
        if leap_mgr.update():
            target_pos = leap_mgr.palms[0]

        input_angles = robot_controller.update(robot_system.get_angles_rad(), target_pos).input_angles_rad
        robot_system.input_angles(input_angles)
        robot_system.input_angles(input_angles)

        robot_visualizer.add_point(target_pos[0, 0],
                                   target_pos[1, 0],
                                   target_pos[2, 0]).add_robot(robot_system.joint_pos).draw()



