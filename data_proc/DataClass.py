#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv 
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import numpy as np
import glob
import math

from scipy import stats, optimize
from geo import Vector, Quaternion
from scipy import interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.graphics.gofplots import qqplot
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

class Utils:
    def plot_user_time_to_target(self, data, user):
        colour = ['r', 'g', 'b']
        c = 0
        for seq in data.keys():
            for dset in data[seq]:
                if user in dset.file:
                    print(dset.file, user)
                    diff = []
                    for i in range(len(dset.target_indices) - 1):
                        if dset.data['tz'][dset.target_indices[i + 1]] == 0 or dset.data['tz'][dset.target_indices[i]] == 0:
                            continue
                        diff.append(dset.data['time'][dset.target_indices[i + 1]] - dset.data['time'][dset.target_indices[i]])
                    plt.plot(diff, colour[c], label=str(seq + ' average: ' + str(np.average(np.array(diff)))))
                    plt.plot((0, len(diff)), (np.average(np.array(diff)), np.average(np.array(diff))), colour[c])
                    c = (c + 1) % 3
        plt.legend()
        plt.grid()
        plt.show()
        
    def get_angles(self, q):

        q.normalise()
        # Rotate Tango orientation to fall onto the world axis system
        orientation = q
        orientation.normalise()

        forward_vector = Vector(0.0, 0.0, 1.0)
        forward_facing_vector = forward_vector.rotate_vector(orientation)
        forward_facing_vector.normalise()

        tango_roll = Vector(forward_facing_vector.x, forward_facing_vector.y, 0)
        tango_roll.normalise()
        tango_pitch = Vector(0, forward_facing_vector.y, forward_facing_vector.z)
        tango_pitch.normalise()
        tango_yaw = Vector(forward_facing_vector.x, 0, forward_facing_vector.z)
        tango_yaw.normalise()
        
        roll_axis = Vector(1, 1, 0)
        roll_axis.normalise()
        pitch_axis = Vector(0, 1, 1)
        pitch_axis.normalise()
        yaw_axis = Vector(1, 0, 1)
        yaw_axis.normalise()

        roll = math.atan2(forward_facing_vector.y, forward_facing_vector.x)
        pitch = math.atan2(forward_facing_vector.y, forward_facing_vector.z)
        yaw = math.atan2(forward_facing_vector.x, forward_facing_vector.z)
                    
        # Test to see if rotation is positive or negative
        test = tango_roll.cross_product(roll_axis)
            
        test = tango_pitch.cross_product(pitch_axis)
            
        test = tango_yaw.cross_product(yaw_axis)

        return [roll-math.pi/2, pitch-math.pi/2, yaw]

class Data:
    def __init__(self, _file, test, seq = None):
        self.file = _file
        self.test = test
        self.seq = seq

        if test == 'spatial':
            self.data = {'correct': [], 'user': [], 'distance': []}
        if test == 'target':
            self.data = {'time': [], 'x': [], 'y': [], 'z': [], 'qx': [], 'qy': [], 'qz': [], 'qw': [], 'obs_distance': [], 'gain': [], 'pitch': [], 'tx': [], 'ty': [], 'tz': []}
        if test == 'tone':
            self.data = {'correct': [], 'user': [], 'first_tone': [], 'second_tone': []}
        if test == 'limit':
            self.data = {'hi': [], 'low': []}

        self.populate_data()

    def populate_data(self):
        if self.seq != None:
            dirs = glob.glob(self.file)
            for _dir in dirs:
                if 'day7' in _dir:
                    reader = csv.reader(open(_dir, 'r'), delimiter="'")
                else:
                    reader = csv.reader(open(_dir, 'r'), delimiter=",")

                for line in reader:
                    # print(_dir, line)
                    self.data['time'].append(float(line[0]))
                    self.data['x'].append(float(line[1]))
                    self.data['y'].append(float(line[2]))
                    self.data['z'].append(float(line[3]))
                    self.data['qx'].append(float(line[4]))
                    self.data['qy'].append(float(line[5]))
                    self.data['qz'].append(float(line[6]))
                    self.data['qw'].append(float(line[7]))
                    if 'day7' in _dir or 'day8' in _dir or 'day9' in _dir or 'day10' in _dir:
                        self.data['pitch'].append(float(line[8]))
                        self.data['tx'].append(-float(line[9]))
                        self.data['ty'].append(float(line[10]))
                        self.data['tz'].append(float(line[11]))
                        self.data.pop('obs_distance', None)
                        self.data.pop('gain', None)
                    else:
                        self.data['obs_distance'].append(float(line[8]))
                        self.data['gain'].append(float(line[10]))
                        self.data['pitch'].append(float(line[11]))
                        self.data['tx'].append(float(line[12]))
                        self.data['ty'].append(float(line[13]))
                        self.data['tz'].append(float(line[14]))

            self.target_indices = [0]
            for i in range(1, len(self.data['x'])):
                # Correct for bug in Android app code...
                if self.data['ty'][i] < 0:
                    self.data['ty'][i] += 0.75
                if self.data['tx'][i] != self.data['tx'][i - 1] and self.data['tz'][i - 1]  == -2 and self.data['tz'][i] == -2:
                    self.target_indices.append(i - 1)
            # Cut out first element since they are the zero start elements
            self.target_indices = self.target_indices[1:]
        else:
            reader = csv.reader(open(self.file, 'r'))
        
            for line in reader:
                if self.test == 'spatial' and len(line) != 0:
                    self.data['correct'].append(line[0])
                    self.data['user'].append(line[1])
                    if len(line) > 3:
                        self.data['distance'].append(float(line[2]))

                if self.test == 'tone' and len(line) != 0:
                    self.data['correct'].append(line[0])
                    self.data['user'].append(line[1])
                    if len(line) > 3:
                        self.data['first_tone'].append(float(line[2]))
                        self.data['second_tone'].append(float(line[3]))

                if self.test == 'limit' and len(line) != 0:
                    self.data['hi'].append(max(map(float, line[:-1])))
                    self.data['low'].append(min(map(float, line[:-1])))

    def get(self, idx):
        return {d: self.data[d][idx] for d in self.data.keys()}

    def plot_spatial(self):
        for i in range(len(self.data['distance'])):
            if self.data['correct'][i] == self.data['user'][i]:
                plt.plot(i, self.data['distance'][i], 'go')
            else:
                plt.plot(i, self.data['distance'][i], 'rx')

        plt.plot(self.data['distance'], 'b--')
        plt.grid()
        plt.show()

    def plot_tone(self):
        diff = []
        for i in range(len(self.data['correct'])):
            diff.append(self.data['first_tone'][i] - self.data['second_tone'][i])
            if self.data['correct'][i] == self.data['user'][i]:
                plt.plot(i, diff[i], 'go')
            else:
                plt.plot(i, diff[i], 'rx')

        plt.plot(diff, 'b--')
        plt.grid()
        plt.show()

    def plot_limit(self):
        for i in range(len(self.data['hi'])):
            point = [self.data['hi'][i], self.data['low'][i]]
            plt.plot([i, i], point)
        plt.show()

    def plot_target(self, param = None):
        q = np.array([self.data['qx'], self.data['qy'], self.data['qz'], self.data['qw']])
        qi = np.array([-q[0], -q[1], -q[2], q[3]])
        f = np.array([0, 1, 0])

        c = ['r', 'c', 'b', 'y', 'k', 'g']
        i = 0
        if param == None:
            for index in self.target_indices:
                vector = self.rotate_vector(f, q[:, index])
                fact = 2 / vector[-1]
                vector *= fact

                plt.plot(self.data['tx'][index], self.data['ty'][index], c[i] + 'o')
                plt.plot(vector[0], vector[1], c[i] + 'x')
                i = (i + 1) % len(c)

            plt.show()

        else:
            meas = []
            for x in self.data[param]:
                meas.append(x)
            plt.plot(meas)
            plt.show()

    def plot_target_error(self):
        err_l = []

        for index in self.target_indices:
            tango_orientation = Quaternion(x = self.data['qx'][index], y = self.data['qy'][index], z = self.data['qz'][index], w = self.data['qw'][index])
            tango_orientation.normalise()

            # Rotate the Tango's orientation to fall in the world axis
            rotate = Quaternion(vector=Vector(1, 0, 0), angle=-math.pi/2)
            rotate.normalise()
            tango_orientation = tango_orientation.multiply(rotate)
            tango_orientation.normalise()

            tango_forward_vector = Vector(0.0, 0.0, -1.0)
            tango_forward_facing_vector = tango_forward_vector.rotate_vector(tango_orientation)
            tango_forward_facing_vector.normalise()

            vector_to_target = Vector(self.data['tx'][index] - self.data['x'][index], self.data['ty'][index] - self.data['z'][index], self.data['tz'][index] - (-1 * self.data['y'][index]))
            vector_to_target.normalise()


            # tango = Vector(tango_forward_facing_vector.x, 0, tango_forward_facing_vector.z)
            tango = Vector(0, tango_forward_facing_vector.y, tango_forward_facing_vector.z)
            tango.normalise()
            # target = Vector(vector_to_target.x, 0, vector_to_target.z)
            target = Vector(0, vector_to_target.y, vector_to_target.z)
            target.normalise()
            angle = tango.inv_dot_product(target)
            
            # Test to see if rotation is positive or negative
            test = tango.cross_product(target)
            if test.x < 0:
                angle = -angle
            err_l.append(math.degrees(angle))

        plt.plot(err_l)
        plt.show()

    def rotate_vector(self, v, q):
        x = (1 - 2 * q[1] * q[1] - 2 * q[2] * q[2]) * v[0] +\
                2 * (q[0] * q[1] + q[3] * q[2]) * v[1] +\
                2 * (q[0] * q[2] - q[3] * q[1]) * v[2]
        y = 2 * (q[0] * q[1] - q[3] * q[2]) * v[0] +\
                (1 - 2 * q[0] * q[0] - 2 * q[2] * q[2]) * v[1] +\
                2 * (q[1] * q[2] + q[3] * q[0]) * v[2]
        z = 2 * (q[0] * q[2] + q[3] * q[1]) * v[0] +\
                2 * (q[1] * q[2] - q[3] * q[0]) * v[1] +\
                (1 - 2 * q[0] * q[0] - 2 * q[1] * q[1]) * v[2]

        return np.array([x, y, z])


class MassData:
    def __init__(self, data, seq=None):
        self.data = data
        self.seq = seq

    def get_tango_angle(self, dset, index, tilt=True):
        tango_orientation = Quaternion(x=dset.data['qx'][index], y=dset.data['qy'][index], z=dset.data['qz'][index], w=dset.data['qw'][index])
        tango_orientation.normalise()

        # Rotate Tango orientation to fall onto the world axis system
        rotate = Quaternion(vector=Vector(1, 0, 0), angle=-math.pi/2)
        rotate.normalise()
        tango_orientation = tango_orientation.multiply(rotate)
        tango_orientation.normalise()
        # print('Qx {}'.format(tango_orientation.x))
        # print('Qy {}'.format(tango_orientation.y))
        # print('Qz {}'.format(tango_orientation.z))
        # print('Qw {}'.format(tango_orientation.w))

        tango_forward_vector = Vector(0.0, 0.0, -1.0)
        tango_forward_facing_vector = tango_forward_vector.rotate_vector(tango_orientation)
        tango_forward_facing_vector.normalise()
        # print('x {} {}'.format(tango_forward_facing_vector.x, dset.data['tx'][index]))
        # print('y {} {}'.format(tango_forward_facing_vector.y, dset.data['ty'][index]))
        # print('z {} {}'.format(tango_forward_facing_vector.z, dset.data['tz'][index]))

        vector_to_target = Vector(dset.data['tx'][index] - dset.data['x'][index], dset.data['ty'][index] - dset.data['z'][index], dset.data['tz'][index] - dset.data['y'][index])
        vector_to_target.normalise()
        rotate = Quaternion(vector=Vector(0, 0, 1), angle=math.pi)
        rotate.normalise()
        vector_to_target = vector_to_target.rotate_vector(rotate)
        vector_to_target.normalise()

        if tilt:
            tango = Vector(0, tango_forward_facing_vector.y, tango_forward_facing_vector.z)
            tango.normalise()
            target = Vector(0, vector_to_target.y, vector_to_target.z)
            target.normalise()

            angle = tango.inv_dot_product(target)
            
            # Test to see if rotation is positive or negative
            test = tango.cross_product(target)
            if test.x < 0:
                angle = -angle

        else:
            tango = Vector(tango_forward_facing_vector.x, 0, tango_forward_facing_vector.z)
            tango.normalise()
            target = Vector(vector_to_target.x, 0, vector_to_target.z)
            target.normalise()

            angle = tango.inv_dot_product(target)
            
            # Test to see if rotation is positive or negative
            test = tango.cross_product(target)
            if test.y < 0:
                angle = -angle

        return angle

    def conv_data_to_angles(self, dset, tilt=True):
        angles = []
        for index in range(len(dset.data['qx'])):
            q1 = Quaternion(x = dset.data['qx'][index], y = dset.data['qy'][index], z = dset.data['qz'][index], w = dset.data['qw'][index])
            q1.normalise()
            tmp = q1.y
            q1.y = q1.z
            q1.z = -tmp

            test = q1.x*q1.y + q1.z*q1.w
            if test > 0.499:
		# attitude = Math.PI/2;
                if tilt:
                    X = 0
                    angle = X
                else:
                    Y = 2 * math.atan2(q1.x,q1.w)
                    angle = Y
            elif test < -0.499:
                if tilt:
                    X = 0
                    angle = X
                else:
                    Y = -2 * math.atan2(q1.x,q1.w)
                    angle = Y

            else:
                sqx = q1.x*q1.x
                sqy = q1.y*q1.y
                sqz = q1.z*q1.z
                if tilt:
                    X = math.atan2(2*q1.x*q1.w-2*q1.y*q1.z , 1 - 2*sqx - 2*sqz)
                    angle = X-math.pi/2
                    if angle < -math.pi:
                        angle += math.pi
                else:
                    Y = math.atan2(2*q1.y*q1.w-2*q1.x*q1.z , 1 - 2*sqy - 2*sqz)
                    angle = Y

            if angle > math.pi/2:
                angle -= math.pi
            elif angle < -math.pi/2:
                angle += math.pi

            angles.append(angle)

        return angles

    def print_errors(self):
        import csv
        for key in ['lo', 'med', 'hi']:
            writer = csv.writer(open(key + '.csv', 'w'))
            for dset in self.data[key]:
                for index in dset.target_indices:
                    pan = self.get_tango_angle(dset, index, tilt=False)
                    tilt = self.get_tango_angle(dset, index, tilt=True)
                    writer.writerow([pan, tilt])

    def plot_error_tilt(self):
        t_avgs = {}
        t_stds = {}
        p_avgs = {}
        p_stds = {}
        for key in ['lo', 'med', 'hi']:
            f, axarr = plt.subplots(1, 1)
            x = self.data[key]

            err_l = []
            pans = []
            tilts = []

            t_avgs[key] = []
            p_avgs[key] = []
            p_stds[key] = []
            t_stds[key] = []

            for dset in x:
                t_err = []
                p_err = []
                for index in dset.target_indices:
                    if len(dset.data['time']) == 0 or dset.data['x'][index] == 0:
                        print('Came across an empty dataset. Skipping.')
                        continue

                    tango_orientation = Quaternion(x=dset.data['qx'][index], y=dset.data['qy'][index], z=dset.data['qz'][index], w=dset.data['qw'][index])
                    tango_orientation.normalise()

                    # Rotate Tango orientation to fall onto the world axis system
                    rotate = Quaternion(vector=Vector(1, 0, 0), angle=-math.pi/2)
                    rotate.normalise()
                    tango_orientation = tango_orientation.multiply(rotate)
                    tango_orientation.normalise()

                    tango_forward_vector = Vector(0.0, 0.0, -1.0)
                    tango_forward_facing_vector = tango_forward_vector.rotate_vector(tango_orientation)
                    tango_forward_facing_vector.normalise()

                    angle = self.get_tango_angle(dset, index, tilt=True)
                    tilt = self.get_tango_angle(dset, index, tilt=True)
                    if tilt < -math.pi/2:
                        tilt += math.pi
                    elif tilt > math.pi/2:
                        tilt -= math.pi
                    tilts.append(tilt)

                    pan = self.get_tango_angle(dset, index, tilt=False)
                    if pan < -math.pi/2:
                        pan += math.pi
                    elif pan > math.pi/2:
                        pan -= math.pi
                    pans.append(pan)

                    err_l.append(math.degrees(angle))
                    t_err.append(tilt)
                    p_err.append(pan)
                if len(t_err) > 0:
                    # t_avgs[key].append(np.average(t_err))
                    t_avgs[key].append(np.median(t_err))
                    t_stds[key].append(np.std(t_err))
                if len(p_err) > 0:
                    # p_avgs[key].append(np.average(p_err))
                    p_avgs[key].append(np.median(p_err))
                    p_stds[key].append(np.std(p_err))
            # plt.hist(avgs[key], bins=20)
            # plt.show()
            # plt.hist(stds[key], bins=20)
            # plt.show()
            err_l = np.array(err_l)
            avg = np.average(err_l)
            std = np.std(err_l)

            # Discard outliers
            # err_l = err_l[abs(err_l - avg) < 2 * std]
            # avg = math.degrees(np.average(np.array(tilts)))
            # std = math.degrees(np.std(np.array(tilts)))

            print(key)
            print(avg, std)

            axarr.plot([-math.pi/2, math.pi/2], [0, 0], 'k--')
            axarr.plot([0, 0], [-math.pi/2, math.pi/2], 'k--')
            axarr.hist2d(pans, tilts, bins=100, range=[[-math.pi/2, math.pi/2], [-math.pi/2, math.pi/2]], cmap='Greys')
            axarr.set_xlabel('Pan Error [rad]', fontsize=28)
            axarr.set_ylabel('Tilt Error [rad]', fontsize=28)
            axarr.tick_params(axis='both', which='major', labelsize=22)
            axarr.set_aspect('equal')
            # axarr.set_xlim([-math.pi/2, math.pi/2])
            # axarr.set_ylim([-math.pi/2, math.pi/2])

            plt.suptitle(key, fontsize=32)
            plt.show()

            # tilt_box_err.append(tilts)
            # pan_box_err.append(pans)
        b0 = plt.boxplot([p_avgs['lo'], t_avgs['lo']], positions=[0.8, 1.2], widths=0.3, showfliers=False, showmeans=True, patch_artist=True)
        b1 = plt.boxplot([p_avgs['med'], t_avgs['med']], positions=[1.8, 2.2], widths=0.3, showfliers=False, showmeans=True, patch_artist=True)
        b2 = plt.boxplot([p_avgs['hi'], t_avgs['hi']], positions=[2.8, 3.2], widths=0.3, showfliers=False, showmeans=True, patch_artist=True)
        plt.xlim([0, 3.8])
        b0['boxes'][0].set(hatch='/')
        b0['boxes'][0].set(fill=False)
        b0['boxes'][0].set_label('Pan')
        b0['boxes'][1].set(fill=False)
        b1['boxes'][0].set(hatch='/')
        b1['boxes'][0].set(fill=False)
        b1['boxes'][1].set(fill=False)
        b2['boxes'][0].set(hatch='/')
        b2['boxes'][0].set(fill=False)
        b2['boxes'][1].set(fill=False)
        plt.grid()
        plt.ylabel('Angle Error [rad]', fontsize=44)
        plt.xticks([1, 2, 3], ['lo', 'med', 'hi'], fontsize=44)
        plt.yticks(fontsize=38)
        plt.legend([b0["boxes"][0], b0["boxes"][1]], ['Pan', 'Tilt'], loc='lower left', fontsize=38)
        plt.ylim([-1.2, 0.8])
        plt.show()

        b0 = plt.boxplot([p_stds['lo'], t_stds['lo']], positions=[0.8, 1.2], widths=0.3, showfliers=False, showmeans=True, patch_artist=True)
        b1 = plt.boxplot([p_stds['med'], t_stds['med']], positions=[1.8, 2.2], widths=0.3, showfliers=False, showmeans=True, patch_artist=True)
        b2 = plt.boxplot([p_stds['hi'], t_stds['hi']], positions=[2.8, 3.2], widths=0.3, showfliers=False, showmeans=True, patch_artist=True)
        plt.xlim([0, 3.8])
        b0['boxes'][0].set(hatch='/')
        b0['boxes'][0].set(fill=False)
        b0['boxes'][0].set_label('Pan')
        b0['boxes'][1].set(fill=False)
        b1['boxes'][0].set(hatch='/')
        b1['boxes'][0].set(fill=False)
        b1['boxes'][1].set(fill=False)
        b2['boxes'][0].set(hatch='/')
        b2['boxes'][0].set(fill=False)
        b2['boxes'][1].set(fill=False)
        plt.grid()
        plt.ylabel('Angle Error [rad]', fontsize=44)
        plt.xticks([1, 2, 3], ['lo', 'med', 'hi'], fontsize=44)
        plt.yticks(fontsize=38)
        plt.legend([b0["boxes"][0], b0["boxes"][1]], ['Pan', 'Tilt'], loc='lower left', fontsize=38)
        plt.ylim([0.0, 0.7])
        plt.show()

        # b0 = plt.boxplot([pan_box_err[0], tilt_box_err[0]], positions=[0.8, 1.2], widths=0.3, showfliers=False, showmeans=True, patch_artist=True)
        # b1 = plt.boxplot([pan_box_err[1], tilt_box_err[1]], positions=[1.8, 2.2], widths=0.3, showfliers=False, showmeans=True, patch_artist=True)
        # b2 = plt.boxplot([pan_box_err[2], tilt_box_err[2]], positions=[2.8, 3.2], widths=0.3, showfliers=False, showmeans=True, patch_artist=True)
        # b0['boxes'][0].set(hatch='/')
        # b0['boxes'][0].set(fill=False)
        # b0['boxes'][0].set_label('Pan')
        # b0['boxes'][1].set(fill=False)
        # b1['boxes'][0].set(hatch='/')
        # b1['boxes'][0].set(fill=False)
        # b1['boxes'][1].set(fill=False)
        # b2['boxes'][0].set(hatch='/')
        # b2['boxes'][0].set(fill=False)
        # b2['boxes'][1].set(fill=False)
        # plt.grid()
        # plt.ylabel('Angle Error [rad]', fontsize=44)
        # plt.xticks([1, 2, 3], ['lo', 'med', 'hi'], fontsize=44)
        # plt.xlim([0.5, 3.5])
        # plt.yticks(fontsize=38)
        # plt.legend([b0["boxes"][0], b0["boxes"][1]], ['Pan', 'Tilt'], loc='lower left', fontsize=38)
        # plt.show()

        # print stats.f_oneway(box_err[0], box_err[1])#, box_err[2])
        # print stats.mstats.kruskalwallis(box_err[0], box_err[1], box_err[2])
        # print stats.median_test(box_err[0], box_err[1], box_err[2])

        for key in p_avgs.keys():
            print(key, 'medians')
            print(stats.shapiro(p_avgs[key]), stats.shapiro(t_avgs[key]))
            # print(stats.normaltest(p_avgs[key]), stats.normaltest(t_avgs[key]))
            # print(stats.anderson(p_avgs[key]), stats.anderson(t_avgs[key]))
            # print(key, 'std')
            # print(stats.shapiro(p_stds[key]), stats.shapiro(t_stds[key]))
            # print(stats.normaltest(p_stds[key]), stats.normaltest(t_stds[key]))
            # print(stats.anderson(p_stds[key]), stats.anderson(t_stds[key]))

            # qqplot(np.array(p_avgs[key]), line='s', color='b')
            # qqplot(np.array(t_avgs[key]), line='s', color='r')
            # plt.show()
        print(stats.f_oneway(t_avgs['lo'], t_avgs['med'], t_avgs['hi']))
        print(stats.f_oneway(p_avgs['lo'], p_avgs['med'], p_avgs['hi']))

    def rad_to_hz(self, rad, key):
        if key == 'lo':
            hi = 10
            lo = 8
        if key == 'med':
            hi = 11
            lo = 7
        if key == 'hi':
            hi = 12
            lo = 6

        m = (hi - lo) / math.pi
        c = (hi - math.pi) / 2 * m

        return m * rad + c

    def plot_error_pan(self):
        box_pan_err = []
        box_tilt_err = []
        for key in ['lo', 'med', 'hi']:
            i = 0
            
            pan_targets = []
            pan_users = []
            tilt_targets = []
            tilt_users = []

            pan_err = []
            tilt_err = []

            for dset in self.data[key]:
                u_user_pan = []
                t_user_tilt = []
                u_user_tilt = []
                t_user_pan = []
                for index in dset.target_indices:
                    if len(dset.data['time']) == 0 or dset.data['x'][index] == 0:
                        print('Came across an empty dataset. Skipping.')
                        continue
                    tango_orientation = Quaternion(x=dset.data['qx'][index], y=dset.data['qy'][index], z=dset.data['qz'][index], w=dset.data['qw'][index])
                    tango_orientation.normalise()

                    # Rotate Tango orientation to fall onto the world axis system
                    rotate = Quaternion(vector=Vector(1, 0, 0), angle=-math.pi/2)
                    rotate.normalise()
                    tango_orientation = tango_orientation.multiply(rotate)
                    tango_orientation.normalise()

                    tango_forward_vector = Vector(0.0, 0.0, -1.0)
                    tango_forward_facing_vector = tango_forward_vector.rotate_vector(tango_orientation)
                    tango_forward_facing_vector.normalise()

                    norm = dset.data['tz'][index] / tango_forward_facing_vector.z
                    tango_forward_facing_vector.x *= norm
                    tango_forward_facing_vector.y *= norm
                    tango_forward_facing_vector.z *= norm

                    # print('x {} y {} z {}'.format(tango_forward_facing_vector.x, tango_forward_facing_vector.y, tango_forward_facing_vector.z))

                    target_pan = math.atan2(dset.data['tx'][index], -dset.data['tz'][index])
                    user_pan = math.atan2(tango_forward_facing_vector.x, -tango_forward_facing_vector.z) 

                    # print(target_pan - user_pan)

                    target_tilt = math.atan2(dset.data['ty'][index], -dset.data['tz'][index])
                    user_tilt = math.atan2(tango_forward_facing_vector.y, -tango_forward_facing_vector.z) 

                    pan_targets.append(target_pan)
                    pan_users.append(-user_pan)

                    tilt_targets.append(target_tilt)
                    tilt_users.append(-user_tilt)

                    t_user_pan.append(target_pan)
                    u_user_pan.append(-user_pan)
                    t_user_tilt.append(target_tilt)
                    u_user_tilt.append(-user_tilt)

                    pan = self.get_tango_angle(dset, index, tilt=False)
                    if pan < -math.pi/2:
                        pan += math.pi
                    elif pan > math.pi/2:
                        pan -= math.pi
                    pan_err.append(pan)

                    tilt = self.get_tango_angle(dset, index, tilt=True)
                    if tilt < -math.pi/2:
                        tilt += math.pi
                    elif tilt > math.pi/2:
                        tilt -= math.pi
                    tilt_err.append(tilt)

                # f, axarr = plt.subplots(1, 2)
                # f.suptitle(dset.file.split('/')[4])
                # axarr[0].plot(u_user_pan, t_user_pan, 'x')
                # axarr[0].set_xlim([-math.pi/2, math.pi/2])
                # axarr[0].set_ylim([-math.pi/2, math.pi/2])
                # axarr[0].set_xlabel('User Pan')
                # axarr[0].set_ylabel('Target Pan')
                # axarr[0].set_title('Pan')

                # axarr[1].plot(u_user_tilt, t_user_tilt, 'x')
                # axarr[1].set_xlim([-math.pi/2, math.pi/2])
                # axarr[1].set_ylim([-math.pi/2, math.pi/2])
                # axarr[1].set_xlabel('User Tilt')
                # axarr[1].set_ylabel('Target Tilt')
                # axarr[1].set_title('Tilt')

                # plt.show()

            # print([t - u for t, u in zip(target_pan, user_pan)])
            f, axarr = plt.subplots(1, 2)
            axarr[0].plot(pan_users, pan_targets, 'x')
            axarr[0].set_xlabel('User Pan')
            axarr[0].set_ylabel('Target Pan')
            axarr[0].set_xlim([-math.pi/2, math.pi/2])
            axarr[0].set_ylim([-math.pi/2, math.pi/2])

            pan_err = np.array(pan_err)
            pan_avg = np.average(pan_err)
            pan_std = np.std(pan_err)

            tilt_err = np.array(tilt_err)
            tilt_avg = np.average(tilt_err)
            tilt_std = np.std(tilt_err)

            # Discard outliers
            pan_err = pan_err[abs(pan_err - pan_avg) < 2 * pan_std]
            pan_avg = np.average(pan_err)
            pan_std = np.std(pan_err)

            tilt_err = tilt_err[abs(tilt_err - tilt_avg) < 2 * tilt_std]
            tilt_avg = np.average(tilt_err)
            tilt_std = np.std(tilt_err)

            weights = np.ones_like(pan_err) / float(len(pan_err))
            hist, bins, patches = axarr[1].hist(pan_err, 50, weights=weights)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axarr[1].bar(center, hist, align='center', width=width)
            axarr[1].grid()
            # axarr[1].set_title('Pan Error: ' + key + ' config')
            axarr[1].set_xlabel('Pan Angle Angle Estimation Error [rad]', fontsize=26)
            axarr[1].set_ylabel('Number of Samples [normalised]', fontsize=26)
            axarr[1].set_xlim([-math.pi / 2, math.pi / 2])
            axarr[1].tick_params(axis='both', which='major', labelsize=20)
            axarr[1].set_ylim([0, 0.1])

            # Plot normal distribution for the lulz
            # t = np.linspace(-math.pi / 2, math.pi / 2, 5000)
            # y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp( - (t- avg)**2 / (2 * std**2))
            # axarr[1].plot(t, y / np.linalg.norm(y), linewidth=2, color='r')

            # axarr[0].set_xlabel('True Pan [rad]', fontsize=26)
            # axarr[0].set_ylabel("User's Pan [rad]", fontsize=26)
            # axarr[0].grid()
            # axarr[0].plot(targets, users, 'x')
            # axarr[0].plot(targets, np.array(targets) * m + c)
            # axarr[0].tick_params(axis='both', which='major', labelsize=20)
            # axarr[0].set_ylim([-3, 3])
            # axarr[0].set_xlim([-0.7, 0.7])

            print(key)
            print(np.average(np.abs(tilt_err)))
            print(tilt_avg, tilt_std)
            # print stats.pearsonr(targets, users)
            # print(pan_targets, pan_users)
            print(stats.spearmanr(pan_targets, pan_users))
            print(stats.spearmanr(tilt_targets, tilt_users))

            i += 1

            plt.suptitle('Pan Results for: ' + key, fontsize=32)
            plt.show()
            box_pan_err.append(pan_err)
            box_tilt_err.append(tilt_err)
        plt.boxplot(box_tilt_err, showfliers=False, positions=[0.8, 1.8, 2.8], widths=0.3, showmeans=True)
        plt.boxplot(box_pan_err, showfliers=False, positions=[1.2, 2.2, 3.2], widths=0.3, showmeans=True)
        plt.grid()
        plt.ylabel('Pan Estimation Error [rad]', fontsize=26)
        plt.title('Boxplots for the Pan Estimation Error', fontsize=32)
        plt.xticks([1, 2, 3], ['lo', 'med', 'hi'], fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

        # print(stats.friedmanchisquare(box_err[0], box_err[1], box_err[2]))
        # print(stats.f_oneway(box_err[0], box_err[1], box_err[2]))
        # print(stats.mstats.kruskalwallis(box_err[0], box_err[1], box_err[2]))
        # print(stats.median_test(box_err[0], box_err[1], box_err[2]))

    def analyse_time2(self):
        def func(x, m, c):
            return m * np.log2(x) + c

        ids = {'hi': [], 'lo': [], 'med': []}
        ips = {'hi': [], 'lo': [], 'med': []}
        times = {'hi': [], 'lo': [], 'med': []}
        wes = {'hi': [], 'lo': [], 'med': []}
                
        nbins = 20
        interval = math.pi/nbins/2

        for key in ids.keys():
            for dset in self.data[key]:
                # user_split_distances = {intervals[i]: [] for i in range(nbins)}
                if len(dset.data['time']) == 0:
                    continue
                user_time_d = []
                user_target_d = []
                user_error_d = []
                for i in range(len(dset.target_indices) - 1):
                    time = dset.data['time'][dset.target_indices[i]] - dset.data['time'][dset.target_indices[i - 1]]
                    if time < 2 or time > 40:
                        continue
                    user_time_d.append(time)

                    pan = abs(self.get_tango_angle(dset, dset.target_indices[i], tilt=False))
                    tilt = abs(self.get_tango_angle(dset, dset.target_indices[i], tilt=True))

                    if pan > math.pi / 2:
                        pan -= math.pi / 2
                    if tilt > math.pi / 2:
                        tilt -= math.pi / 2

                    pan_d = math.sqrt(2 * 2 ** 2 - 2 * 2 * 2 * math.cos(pan))
                    tilt_d = math.sqrt(2 * 2 ** 2 - 2 * 2 * 2 * math.cos(tilt))
                    error = math.sqrt(pan_d ** 2 + tilt_d ** 2)
                    error = math.acos((8 - error ** 2) / 8.0)
                    user_error_d.append(error)

                    target = math.sqrt((dset.data['tx'][dset.target_indices[i]] - dset.data['tx'][dset.target_indices[i - 1]]) ** 2 + (dset.data['ty'][dset.target_indices[i]] - dset.data['ty'][dset.target_indices[i - 1]]) ** 2)
                    target = math.acos((8 - target**2) / 8.0)
                    user_target_d.append(target)

                we = 4.133 * np.std(user_error_d)
                wes[key].append(we)
                ids[key] += [t/we + 1 for t in user_target_d]
                times[key] += user_time_d

            id_hist, id_edges_ = np.histogram(ids[key], bins=nbins)
            id_edges = []
            interval = (id_edges_[1] - id_edges_[0]) / 2
            for i in range(0, nbins, 1):
                id_edges.append(id_edges_[i] + interval)
            split_distances = {edge: [] for edge in id_edges}

            for t, d in zip(times[key], ids[key]):
                for edge in reversed(id_edges):
                    if d >= edge:
                        split_distances[edge].append(t)
                        continue

            plt.plot(ids[key], times[key], 'rx')
            split_distances = dict(sorted(split_distances.items()))
            medians = np.array([np.mean(x) for x in split_distances.values()])
            id_edges_a = np.array(id_edges)[~np.isnan(medians)]
            medians = medians[~np.isnan(medians)]
            # medians[np.isnan(medians)] = np.mean(medians[~np.isnan(medians)])
            print(medians, id_edges_a)
            popt, pconv = optimize.curve_fit(func, np.array(id_edges_a), medians)
            plt.plot(id_edges_a, medians, 'go')

            boxes = split_distances.values()
            obs = np.array(medians)
            reg = np.log2([d for d in sorted(list(split_distances.keys()))])*popt[0] + popt[1]
            # print(stats.shapiro(obs), stats.shapiro(reg))
            print(obs, reg)
            print(stats.pearsonr(obs, reg))

            x = np.linspace(-1, 4.0, 1000)
            plt.plot(x, np.log2(x) * popt[0] + popt[1], label=r'Log fit', linestyle='--', color='k')
            plt.boxplot(boxes, positions=id_edges, showfliers=False, widths=0.02)

            # plt.tick_params(axis='both', which='major', labelsize=34)
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # No decimal places
            plt.legend(fontsize=34)
            plt.xlabel('ID', fontsize=40)
            plt.ylabel('Time [s]', fontsize=40)
            plt.grid()
            plt.title(key)
            # plt.xlim([-1, interval*(nbins+1)+1])
            # plt.ylim(-20, 43)
            plt.show()

            ips[key] = 1/popt[0]

        print(ips)

        print([stats.shapiro(ids[key]) for key in ids.keys()])
        # print(stats.friedmanchisquare([ips['lo'], ips['med'], ips['hi']]))
        print(stats.f_oneway([ips['lo'], ips['med'], ips['hi']]))
        # print(stats.kruskal(ips['lo'][:l], ips['med'][:l], ips['hi'][:l]))
        # print(stats.friedmanchisquare(1/ms['lo'][:l], 1/ms['med'][:l], 1/ms['hi'][:l]))
        # print(stats.wilcoxon(1/ms['lo'][:l], 1/ms['med'][:l]))
        # print(stats.wilcoxon(1/ms['lo'][:l], 1/ms['hi'][:l]))
        # print(stats.wilcoxon(1/ms['med'][:l], 1/ms['hi'][:l]))
        # print(stats.wilcoxon(ips['lo'], ips['med']))
        # print(stats.mannwhitneyu(ips['lo'][:l], ips['med'][:l]))
        # print(stats.wilcoxon(ips['lo'], ips['hi']))
        # print(stats.mannwhitneyu(ips['lo'][:l], ips['hi'][:l]))
        # print(stats.wilcoxon(ips['med'], ips['hi']))
        # print(stats.mannwhitneyu(ips['med'][:l], ips['hi'][:l]))

    def analyse_time(self):
        def func(x, m, c):
            return m * np.log2(x) + c

        grads = {'hi': [], 'lo': [], 'med': []}
        intercepts = {'hi': [], 'lo': [], 'med': []}
        ids = {'hi': [], 'lo': [], 'med': []}
                
        nbins = 10
        interval = math.pi/nbins/2
        intervals = [float('%0.3f' % (i * interval)) for i in range(1, nbins+1)]

        for key in grads.keys():
            split_distances = {intervals[i]: [] for i in range(nbins)}
            for dset in self.data[key]:
                user_split_distances = {intervals[i]: [] for i in range(nbins)}
                if len(dset.data['time']) == 0:
                    continue
                user_time_d = []
                user_target_d = []
                user_error_d = []
                for i in range(len(dset.target_indices) - 1):
                    time = dset.data['time'][dset.target_indices[i]] - dset.data['time'][dset.target_indices[i - 1]]
                    if time < 2 or time > 40:
                        continue
                    user_time_d.append(time)

                    pan = abs(self.get_tango_angle(dset, dset.target_indices[i], tilt=False))
                    tilt = abs(self.get_tango_angle(dset, dset.target_indices[i], tilt=True))

                    if pan > math.pi / 2:
                        pan -= math.pi / 2
                    if tilt > math.pi / 2:
                        tilt -= math.pi / 2

                    pan_d = math.sqrt(2 * 2 ** 2 - 2 * 2 * 2 * math.cos(pan))
                    tilt_d = math.sqrt(2 * 2 ** 2 - 2 * 2 * 2 * math.cos(tilt))
                    error = math.sqrt(pan_d ** 2 + tilt_d ** 2)
                    error = math.acos((8 - error ** 2) / 8.0)
                    user_error_d.append(error)

                    target = math.sqrt((dset.data['tx'][dset.target_indices[i]] - dset.data['tx'][dset.target_indices[i - 1]]) ** 2 + (dset.data['ty'][dset.target_indices[i]] - dset.data['ty'][dset.target_indices[i - 1]]) ** 2)
                    target = math.acos((8 - target**2) / 8.0)
                    user_target_d.append(target)

                we = 4.133 * np.std(user_error_d)
                # plt.plot([list(user_split_distances.keys())[0]/we + 1]*2, [0, 10], '--b')
                # plt.plot([list(user_split_distances.keys())[1]/we + 1]*2, [0, 10], '--b')
                # plt.plot([list(user_split_distances.keys())[2]/we + 1]*2, [0, 10], '--b')
                # plt.plot([list(user_split_distances.keys())[3]/we + 1]*2, [0, 10], '--b')
                for t, d in zip(user_time_d, user_target_d):
                    if d >= 0.0 and d < intervals[0] + interval / 2:
                        user_split_distances[intervals[0]].append(t)
                    elif d >= intervals[0] + interval / 2 and d < intervals[1] + interval / 2:
                        user_split_distances[intervals[1]].append(t)
                    elif d >= intervals[1] + interval / 2 and d < intervals[2] + interval / 2:
                        user_split_distances[intervals[2]].append(t)
                    elif d >= intervals[2] + interval / 2 and d < intervals[3] + interval / 2:
                        user_split_distances[intervals[3]].append(t)
                    elif d >= intervals[3] + interval / 2 and d < intervals[4] + interval / 2:
                        user_split_distances[intervals[4]].append(t)
                    elif d >= intervals[4] + interval / 2 and d < intervals[5] + interval / 2:
                        user_split_distances[intervals[5]].append(t)
                    elif d >= intervals[5] + interval / 2 and d < intervals[6] + interval / 2:
                        user_split_distances[intervals[6]].append(t)
                    elif d >= intervals[6] + interval / 2 and d < intervals[7] + interval / 2:
                        user_split_distances[intervals[7]].append(t)
                    elif d >= intervals[7] + interval / 2 and d < intervals[8] + interval / 2:
                        user_split_distances[intervals[8]].append(t)
                    elif d >= intervals[8] + interval / 2 and d < intervals[9] + interval / 2:
                        user_split_distances[intervals[9]].append(t)
                    # plt.plot(d/we+1, t, 'xr')

                if len(user_target_d) > 0:
                    id_ = np.array(list(user_split_distances.keys())) / np.abs(we) + 1
                    medians = np.array([np.median(x) for x in user_split_distances.values()])
                    medians[np.isnan(medians)] = np.mean(medians[~np.isnan(medians)])
                    popt, pconv = optimize.curve_fit(func, np.array(id_), medians)
                    grads[key].append(popt[0])
                    intercepts[key].append(popt[1])
                    ids[key].append(np.log2(id_))
                    plt.plot(id_, medians, 'og')
                    print(medians)

                    # plt.show()

                for keyl in split_distances.keys():
                    split_distances[keyl] += user_split_distances[keyl]
                    # print(user_split_distances[keyl])
                    # print(medians.tolist())
                    # split_distances[keyl] += medians.tolist()

            # if len(user_target_d) > 0:
                # x = [inner for outer in split_distances.values() for inner in outer]
                # we = 4.133 * np.std(x)
                # id_ = np.array(list(split_distances.keys())) / np.abs(we) + 1
                # medians = np.array([np.median(x) for x in split_distances.values()])
                # medians[np.isnan(medians)] = np.mean(medians[~np.isnan(medians)])
                # print(medians)
                # popt, pconv = optimize.curve_fit(func, np.array(id_), medians)
                # grads[key].append(popt[0])
                # intercepts[key].append(popt[1])
                # ids[key].append(np.log2(id_))
                # x = np.linspace(-1, 3.0, 1000)
                # plt.plot(id_, medians, 'og')

            boxes = split_distances.values()
            d = np.array([np.array(x) for x in boxes])
            obs = np.array([np.median(x) for x in d])
            reg = np.log2([d/we + 1 for d in list(split_distances.keys())])*np.median(grads[key]) + np.median(intercepts[key])
            # print(stats.shapiro(obs), stats.shapiro(reg))
            print(stats.pearsonr(obs, reg))

            x = np.linspace(-1, 3.0, 1000)
            plt.plot(x, np.log2(x) * np.median(grads[key]) + np.median(intercepts[key]), label=r'Log fit', linestyle='--', color='k')
            box_pos = [x/we + 1 for x in list(split_distances.keys())]
            plt.boxplot(boxes, positions=box_pos, showfliers=False, widths=0.02)

            # plt.tick_params(axis='both', which='major', labelsize=34)
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # No decimal places
            plt.legend(fontsize=34)
            plt.xlabel('ID', fontsize=40)
            plt.ylabel('Time [s]', fontsize=40)
            plt.grid()
            plt.title(key)
            # plt.xlim([-1, interval*(nbins+1)+1])
            # plt.ylim(-20, 43)
            plt.show()

        print([len(grads[key]) for key in grads.keys()])
        ips = {key: [] for key in grads.keys()}
        for key in ips.keys():
            for i in range(len(grads[key])):
                # ips[key] += [id_/(intercepts[key][i] + id_*grads[key][i]) for id_ in ids[key][i]]
                ips[key] += [1/grads[key][i]]

        plt.boxplot([np.array(ips[key]) for key in ips.keys()], positions=[1, 2, 3], showfliers=False, showmeans=False, usermedians=[np.median(ips[key]) for key in ips.keys()], whis=1.5)
        plt.yticks(fontsize=34)
        plt.ylabel('IP', fontsize=40)
        plt.xlim([0, 4.0])
        plt.xticks([1, 2, 3], ['lo', 'med', 'hi'], fontsize=40)
        plt.grid()
        plt.show()

        l = min(len(ips['lo']), len(ips['med']), len(ips['hi']))
        print([stats.shapiro(ids[key][:l]) for key in ids.keys()])
        print(stats.friedmanchisquare(ips['lo'][:l], ips['med'][:l], ips['hi'][:l]))
        # print(stats.kruskal(ips['lo'][:l], ips['med'][:l], ips['hi'][:l]))
        # print(stats.friedmanchisquare(1/ms['lo'][:l], 1/ms['med'][:l], 1/ms['hi'][:l]))
        # print(stats.wilcoxon(1/ms['lo'][:l], 1/ms['med'][:l]))
        # print(stats.wilcoxon(1/ms['lo'][:l], 1/ms['hi'][:l]))
        # print(stats.wilcoxon(1/ms['med'][:l], 1/ms['hi'][:l]))
        print(stats.wilcoxon(ips['lo'][:l], ips['med'][:l]))
        # print(stats.mannwhitneyu(ips['lo'][:l], ips['med'][:l]))
        print(stats.wilcoxon(ips['lo'][:l], ips['hi'][:l]))
        # print(stats.mannwhitneyu(ips['lo'][:l], ips['hi'][:l]))
        print(stats.wilcoxon(ips['med'][:l], ips['hi'][:l]))
        # print(stats.mannwhitneyu(ips['med'][:l], ips['hi'][:l]))

    def plot_avg_time_to_target(self):

        def func(x, m, c):
            # if m < 0:
                # m = abs(m)
            return m * np.log2(x) + c

        wes = {'lo': 1.7, 'med': 1.454, 'hi': 1.584}
        ms = {'hi': [], 'lo': [], 'med': []}
        cs = {'hi': [], 'lo': [], 'med': []}
        ids = {key: [] for key in cs.keys()}
        for key in ['lo', 'med', 'hi']:
            x = self.data[key]
            diffs = []
            errors = []
            distances = []

            for dset in x:
                if len(dset.data['time']) == 0:
                    continue
                user_diff = []
                user_dist = []
                user_errors = []
                for i in range(1, len(dset.target_indices) - 1):
                    diff = dset.data['time'][dset.target_indices[i]] - dset.data['time'][dset.target_indices[i - 1]]
                    pan = self.get_tango_angle(dset, dset.target_indices[i], tilt=False)
                    tilt = self.get_tango_angle(dset, dset.target_indices[i], tilt=True)

                    tilt = abs(tilt)
                    pan = abs(pan)

                    if pan > math.pi / 2:
                        pan -= math.pi / 2
                    if tilt > math.pi / 2:
                        tilt -= math.pi / 2

                    pan_d = math.sqrt(2 * 2 ** 2 - 2 * 2 * 2 * math.cos(pan))
                    tilt_d = math.sqrt(2 * 2 ** 2 - 2 * 2 * 2 * math.cos(tilt))
                    error = math.sqrt(pan_d ** 2 + tilt_d ** 2)
                    error = math.acos((8 - error ** 2) / 8.0)

                    # quat = Quaternion(roll = 0, pitch = tilt, yaw = pan)
                    # quat.normalise()
                    # error = math.acos(quat.w)
                    # if error > math.pi / 2:
                        # error -= math.pi / 2
                    distance = math.sqrt((dset.data['tx'][dset.target_indices[i]] - dset.data['tx'][dset.target_indices[i - 1]]) ** 2 + (dset.data['ty'][dset.target_indices[i]] - dset.data['ty'][dset.target_indices[i - 1]]) ** 2)
                    # Cos rule
                    distance = math.acos((8 - distance ** 2) / 8.0)

                    if diff < 0 or diff > 40:
                        continue
                    diffs.append(diff)
                    errors.append(error)
                    user_errors.append(error)
                    user_diff.append(diff)
                    user_dist.append(distance)
                    distances.append(distance)

                if len(user_dist) > 0:
                    # we = 4.133 * np.std(user_errors)
                    id_ = np.array(user_dist)/wes[key] + 1
                    popt, pconv = optimize.curve_fit(func, id_, np.array(user_diff))
                    m = popt[0]
                    c = popt[1]
                    if m < 0:
                        continue
                    ms[key].append(m)
                    cs[key].append(c)
                    ids[key].append(np.log2(id_))

            we = 4.133 * np.std(errors)
            print(we)
            # ms[key] = np.abs(ms[key])
            ms[key] = np.array(ms[key])
            cs[key] = np.array(cs[key])
            ids[key] = np.array(ids[key])
            # wes[key] = we
            x = np.linspace(-1.0, 3.8, 10000)

            nbins = 10
            # bins, edges = np.histogram(distances, bins=nbins)
            interval = 0.1
            intervals = [float('%0.3f' % (i * interval)) for i in range(1, nbins + 1)]
            split_distances = {intervals[i]: [] for i in range(nbins)}
            # print(ms)

            print(np.min(np.concatenate(ids[key])), np.max(np.concatenate(ids[key])))
            print(np.min(ms[key]), np.max(ms[key]))
            print(np.min(cs[key]), np.max(cs[key]))
            # for diff, dist in zip(diffs, np.array(distances)/we):
            for diff, dist in zip(diffs, np.concatenate(ids[key]).ravel()):
                if dist >= 0.0 and dist < intervals[0] + interval / 2:
                    split_distances[intervals[0]].append(diff)
                elif dist >= intervals[0] + interval / 2 and dist < intervals[1] + interval / 2:
                    split_distances[intervals[1]].append(diff)
                elif dist >= intervals[1] + interval / 2 and dist < intervals[2] + interval / 2:
                    split_distances[intervals[2]].append(diff)
                elif dist >= intervals[2] + interval / 2 and dist < intervals[3] + interval / 2:
                    split_distances[intervals[3]].append(diff)
                elif dist >= intervals[3] + interval / 2 and dist < intervals[4] + interval / 2:
                    split_distances[intervals[4]].append(diff)
                elif dist >= intervals[4] + interval / 2 and dist < intervals[5] + interval / 2:
                    split_distances[intervals[5]].append(diff)
                elif dist >= intervals[5] + interval / 2 and dist < intervals[6] + interval / 2:
                    split_distances[intervals[6]].append(diff)
                elif dist >= intervals[6] + interval / 2 and dist < intervals[7] + interval / 2:
                    split_distances[intervals[7]].append(diff)
                elif dist >= intervals[7] + interval / 2 and dist < intervals[8] + interval / 2:
                    split_distances[intervals[8]].append(diff)
                elif dist >= intervals[8] + interval / 2 and dist < intervals[9] + interval / 2:
                    split_distances[intervals[9]].append(diff)

            boxes = []
            medians = []
            # print(split_distances)
            for keyl in split_distances.keys():
                boxes.append(split_distances[keyl])
                medians.append(np.median(split_distances[keyl]))

            d = np.array([np.array(x) for x in boxes])
            obs = np.array([np.median(x) for x in d])
            reg = np.log2(list(split_distances.keys()))*np.median(ms[key]) + np.median(cs[key])
            # print(stats.shapiro(obs), stats.shapiro(reg))
            # print(stats.spearmanr(obs, reg))
            print(stats.pearsonr(obs, reg))
            # print(obs, reg)

            plt.plot(x, np.log2(x+0.75) * np.median(ms[key]) + np.median(cs[key]), label=r'Log fit', linestyle='--', color='k')
            # for m, c in zip(ms[key], cs[key]):
                # plt.plot(x, np.log2(x) * m + c)

            box_pos = list(split_distances.keys())
            # box_pos = [x for x in box_pos]
            plt.boxplot(boxes, positions=box_pos, showfliers=False, widths=0.05)

            plt.tick_params(axis='both', which='major', labelsize=34)
            plt.legend(fontsize=34)
            plt.xlabel('ID', fontsize=40)
            plt.ylabel('Time [s]', fontsize=40)
            plt.grid()
            plt.xlim([-1.0, interval*(nbins+1)])
            plt.ylim(-10, 30)
            plt.show()
        # Reject outlier gradients
        # derp = {key: ms[key][ms[key] > 1e-6] for key in ms.keys()}
        # ms = derp.copy()
        # derp = {key: ms[key][abs(ms[key] - np.mean(ms[key])) < 3 * np.std(ms[key])] for key in ['lo', 'med', 'hi']}
        # ms = derp.copy()
        # print(ms)

        ips = {key: [] for key in ms.keys()}
        for key in ips.keys():
            for i in range(len(ms[key])):
                ips[key] += [id_/(cs[key][i] + id_*ms[key][i]) for id_ in ids[key][i]]
                # ips[key] += [abs(1/ms[key][i])]
        # ips = {key: np.array(ips[key])[np.array(ips[key])<2*math.pi] for key in ips.keys()}

        # plt.boxplot([1 / np.array(ms[key]) for key in ms.keys()], positions=[1, 2, 3], showfliers=True, showmeans=False)
        # plt.yticks(fontsize=34)
        # plt.ylabel('IP', fontsize=40)
        # plt.xlim([0, 4.0])
        # plt.xticks([1, 2, 3], ['lo', 'med', 'hi'], fontsize=40)
        # plt.grid()
        # plt.show()

        plt.boxplot([np.array(ips[key]) for key in ips.keys()], positions=[1, 2, 3], showfliers=False, showmeans=False, usermedians=[np.median(ips[key]) for key in ips.keys()], whis=1.5)
        plt.yticks(fontsize=34)
        plt.ylabel('IP', fontsize=40)
        plt.xlim([0, 4.0])
        plt.xticks([1, 2, 3], ['lo', 'med', 'hi'], fontsize=40)
        plt.grid()
        plt.show()

        l = min(len(ips['lo']), len(ips['med']), len(ips['hi']))
        print([stats.shapiro(ips[key][:l]) for key in ms.keys()])
        # print([stats.shapiro(ms[key][:l]) for key in ms.keys()])
        print(stats.friedmanchisquare(ips['lo'][:l], ips['med'][:l], ips['hi'][:l]))
        # print(stats.kruskal(ips['lo'][:l], ips['med'][:l], ips['hi'][:l]))
        # print(stats.friedmanchisquare(1/ms['lo'][:l], 1/ms['med'][:l], 1/ms['hi'][:l]))
        # print(stats.wilcoxon(1/ms['lo'][:l], 1/ms['med'][:l]))
        # print(stats.wilcoxon(1/ms['lo'][:l], 1/ms['hi'][:l]))
        # print(stats.wilcoxon(1/ms['med'][:l], 1/ms['hi'][:l]))
        print(stats.wilcoxon(ips['lo'][:l], ips['med'][:l]))
        # print(stats.mannwhitneyu(ips['lo'][:l], ips['med'][:l]))
        print(stats.wilcoxon(ips['lo'][:l], ips['hi'][:l]))
        # print(stats.mannwhitneyu(ips['lo'][:l], ips['hi'][:l]))
        print(stats.wilcoxon(ips['med'][:l], ips['hi'][:l]))
        # print(stats.mannwhitneyu(ips['med'][:l], ips['hi'][:l]))

    # Plot actual path followed (angle), derivative of path, avg speed of search 
    def plot_search_path(self):
        utils = Utils()
        dset = self.data['med'][4]
        rolls = []
        pitches = []
        yaws = []
        
        for i in range(len(dset.data['qx'])):
            angle = utils.get_angles(Quaternion(x=dset.data['qx'][i], y=dset.data['qy'][i], z=dset.data['qz'][i], w=dset.data['qw'][i]))
            rolls.append(math.degrees(angle[0]))
            pitches.append(math.degrees(angle[1]))
            yaws.append(math.degrees(angle[2]))
            
        t_interp = np.linspace(min(dset.data['time']), max(dset.data['time']), 50000)

        s_rolls = interp.UnivariateSpline(dset.data['time'], rolls, s=5)
        s_rolls_deriv = s_rolls.derivative()
        rolls_interp = s_rolls(t_interp)
        rolls_deriv = s_rolls_deriv(t_interp)

        s_pitches = interp.UnivariateSpline(dset.data['time'], pitches, s=5)
        s_pitches_deriv = s_pitches.derivative()
        pitches_interp = s_pitches(t_interp)
        pitches_deriv = s_pitches_deriv(t_interp)

        s_yaws = interp.UnivariateSpline(dset.data['time'], yaws, s=5)
        s_yaws_deriv = s_yaws.derivative()
        yaws_interp = s_yaws(t_interp)
        yaws_deriv = s_yaws_deriv(t_interp)
        
        f, axarr = plt.subplots(3, 2)
        
        axarr[0, 0].plot(t_interp, rolls_interp)
        axarr[0, 1].plot(t_interp, rolls_deriv)
        axarr[1, 0].plot(t_interp, pitches_interp)
        axarr[1, 1].plot(t_interp, pitches_deriv)
        axarr[2, 0].plot(t_interp, yaws_interp)
        axarr[2, 1].plot(t_interp, yaws_deriv)
        
        axarr[0, 0].plot(dset.data['time'], rolls)
        axarr[1, 0].plot(dset.data['time'], pitches)
        axarr[2, 0].plot(dset.data['time'], yaws)
        
        for index in dset.target_indices:
            axarr[0, 0].plot([dset.data['time'][index], dset.data['time'][index]], [min(rolls), max(rolls)])
            axarr[1, 0].plot([dset.data['time'][index], dset.data['time'][index]], [min(pitches), max(pitches)])
            axarr[2, 0].plot([dset.data['time'][index], dset.data['time'][index]], [min(yaws), max(yaws)])

        plt.show()

    # Plot the search trajectory with the points where the user selected the target and the target's actual location
    def plot_target_ack(self):
        dset = self.data['hi'][4]
        print(dset.target_indices)
        xs = []
        ys = []

        for i in range(len(dset.data['x'])):
            pan = self.get_tango_angle(dset, i, tilt=False)
            tilt = self.get_tango_angle(dset, i, tilt=True)
            y = 2.0 * math.tan(tilt)# + dset.data['y'][target_index]
            y = y if dset.data['ty'][i] > 0 else y + 0.75
            x = 2.0 * math.tan(pan)
            ys.append(y + dset.data['ty'][i])
            xs.append(x + dset.data['tx'][i])
            if i in dset.target_indices:
                plt.plot(x + dset.data['tx'][i], y + dset.data['ty'][i], 'rx')
                plt.plot(dset.data['tx'][i], dset.data['ty'][i], 'go')
                plt.plot([x + dset.data['tx'][i], dset.data['tx'][i]], [y + dset.data['ty'][i], dset.data['ty'][i]], 'g--')

        plt.xlim([-2.5, 2.5])
        plt.ylim([-1.5, 1.5])
        plt.show()

    def get_anova(self):
        datasets = {}
        # codes = ['A01', 'A02', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07',
                 # 'C08', 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 
                 # 'E08', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08']
        glob_pan = []
        glob_tilt = []
        for key in ['lo', 'med', 'hi']:
            pan = []
            tilt = []
            for dset in self.data[key]:
                code = dset.file.split('/')[4]
                if code not in datasets.keys():
                    datasets[code] = {}
                datasets[code][key] = {}
                datasets[code][key]['pan'] = []
                datasets[code][key]['tilt'] = []
                pans = []
                tilts = []
                for index in dset.target_indices:
                    if len(dset.data['time']) == 0 or dset.data['x'][index] == 0:
                        print('Came across an empty dataset. Skipping.')
                        continue
                    pans.append(abs(self.get_tango_angle(dset, index, tilt=False)))
                    tilts.append(abs(self.get_tango_angle(dset, index, tilt=True)))
                if len(pans) != 0 and len(tilts) != 0:
                    datasets[code][key]['pan'].append(np.median(pans))
                    datasets[code][key]['tilt'].append(np.median(np.array(tilts)))
                    pan.append(np.median(pans))
                    tilt.append(np.median(tilts))
                    # pans.append(np.median(pan))
                    # tilts.append(np.median(tilt))
            glob_pan.append(pan)
            glob_tilt.append(tilt)
            # print(datasets)
        # pans = np.array()
        # tilt = np.array()
        pans = []
        tilts = []
        for code in datasets.keys():
            pan = []
            tilt = []
            for key in datasets[code].keys():
                if len(datasets[code][key]['tilt']) != 0:
                    pan.append(datasets[code][key]['pan'])
                    tilt.append(datasets[code][key]['tilt'][0])
            tilts.append(tilt)
            pans.append(pan)

        # print(datasets)
        # print([datasets[code]['hi']['pan'][0] if len(datasets[code]['hi']['pan']) != 0 else 0.0 for code in datasets.keys()])
        print(stats.f_oneway([datasets[code]['lo']['pan'][0] if len(datasets[code]['lo']['pan']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['med']['pan'][0] if len(datasets[code]['med']['pan']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['hi']['pan'][0] if len(datasets[code]['hi']['pan']) != 0 else 0.0 for code in datasets.keys()]))
        print(stats.friedmanchisquare([datasets[code]['lo']['pan'][0] if len(datasets[code]['lo']['pan']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['med']['pan'][0] if len(datasets[code]['med']['pan']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['hi']['pan'][0] if len(datasets[code]['hi']['pan']) != 0 else 0.0 for code in datasets.keys()]))

        print(stats.f_oneway([datasets[code]['lo']['tilt'][0] if len(datasets[code]['lo']['tilt']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['med']['tilt'][0] if len(datasets[code]['med']['tilt']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['hi']['tilt'][0] if len(datasets[code]['hi']['tilt']) != 0 else 0.0 for code in datasets.keys()]))
        print(stats.friedmanchisquare([datasets[code]['lo']['tilt'][0] if len(datasets[code]['lo']['tilt']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['med']['tilt'][0] if len(datasets[code]['med']['tilt']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['hi']['tilt'][0] if len(datasets[code]['hi']['tilt']) != 0 else 0.0 for code in datasets.keys()]))
        # print(len([datasets[code]['hi']['tilt'][0] for code in datasets.keys()]))
        # print(len([datasets[code]['med']['tilt'][0] for code in datasets.keys()]))
        # print(len([datasets[code]['lo']['tilt'][0] for code in datasets.keys()]))

        print(stats.wilcoxon([datasets[code]['hi']['tilt'][0] if len(datasets[code]['hi']['tilt']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['med']['tilt'][0] if len(datasets[code]['med']['tilt']) != 0 else 0.0 for code in datasets.keys()]))
        print(stats.wilcoxon([datasets[code]['lo']['tilt'][0] if len(datasets[code]['lo']['tilt']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['med']['tilt'][0] if len(datasets[code]['med']['tilt']) != 0 else 0.0 for code in datasets.keys()]))
        print(stats.wilcoxon([datasets[code]['hi']['tilt'][0] if len(datasets[code]['hi']['tilt']) != 0 else 0.0 for code in datasets.keys()], [datasets[code]['lo']['tilt'][0] if len(datasets[code]['lo']['tilt']) != 0 else 0.0 for code in datasets.keys()]))

        # plt.hist([datasets[code]['lo']['tilt'][0] for code in datasets.keys()], bins=20, normed=1, label='lo')
        # plt.hist([datasets[code]['med']['tilt'][0] if len(datasets[code]['med']['tilt']) != 0 else 0.0 for code in datasets.keys()], bins=20, normed=1, label='med')
        # plt.hist([datasets[code]['hi']['tilt'][0] for code in datasets.keys()], bins=20, normed=1, label='hi')
        # plt.legend()
        # plt.show()
        # print [datasets[code]['hi']['pan'][0] for code in datasets.keys()]
        hatch = {'lo': '/', 'med': 'x', 'hi': 'o'}
        # meh = []
        for key in ['lo', 'med', 'hi']:
            nbins = 5
            # if key == 'med':
                # nbins /= 5
            # if key == 'lo':
                # nbins /= 2
            data = [datasets[code][key]['tilt'][0] if len(datasets[code][key]['tilt']) != 0 else 0.0 for code in datasets.keys()]
            data = [self.rad_to_hz(data[i], key) for i in range(len(data))]
            # meh.append(data)
            # hist, bins = np.histogram(np.log2(data), normed=False, bins=int(nbins))
            hist, bins = np.histogram(data, normed=False, bins=int(nbins))
            width = 0.7 * (bins[1] - bins[0])
            width = 0.05
            center = (bins[:-1] + bins[1:]) / 2
            # 123 is number of samples
            nsamples = len(data)*3
            plt.bar(center, hist/nsamples, align='center', width=width, label=key, fill=False, hatch=hatch[key])
        plt.plot([7, 7], [0, 0.16], '--k', label='75% Threshold')
        # plt.plot([12.9, 12.9], [0, 0.16], '--k', label='75% Threshold')
        # plt.plot([2.97, 2.97], [0, 0.16], linestyle='--', label='95% Confidence Interval', color='r')
        plt.xlabel('Sound Frequency [Hz]', fontsize=40)
        plt.ylabel('Number of Samples [normalised]', fontsize=40)
        plt.tick_params(axis='both', which='major', labelsize=34)
        # plt.xlim([1, 5.0])
        # ax = plt.gca()
        # labels = [1, 2, 4, 8, 16, 32, 64]
        # ax.set_xticklabels(labels)
        plt.legend(fontsize=34)
        plt.show()
