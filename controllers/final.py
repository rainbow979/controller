from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from tdw.tdw_utils import TDWUtils
from tdw.output_data import Images, CameraMatrices
from sticky_mitten_avatar.avatars import Arm
from sticky_mitten_avatar import StickyMittenAvatarController
from sticky_mitten_avatar.util import get_data
from utils import create_tdw, kill_tdw
from queue import Queue
import numpy as np
import pyastar
import time
import cv2
import math
import pickle
from sticky_mitten_avatar.task_status import TaskStatus
from sticky_mitten_avatar.paths import SURFACE_OBJECT_CATEGORIES_PATH

import random

DISTANCE = 0.5
ANGLE = 15.
ANGLE1 = 60.
SCALE = 4
from json import loads
#ANGLE2 = 45.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


fff = open('trans.log', 'w', 10)

DEMO = False

class Nav(StickyMittenAvatarController):
    """
    1. map
    2. global goal
    3. 
    """

    def __init__(self, port: int = 1071, launch_build: bool = False, demo=False):
        """
        :param port: The port number.
        :param launch_build: If True, automatically launch the build.
        """

        super().__init__(port=port, launch_build=launch_build, \
                id_pass=True, demo=DEMO)
        self.demo = demo
        self.max_steps = 1000
        self.map_id = 0
        
    def save_infomation(self, output_directory='.'):
        frame_count = self.frame._frame_count
        print('frame:', frame_count)
        self.frame.save_images(output_directory=output_directory)
        np.save(f'{output_directory}/{frame_count}.npy', self.frame.camera_matrix)
        
        
        '''tmp_map = np.zeros((self.map_shape[0], self.map_shape[1], 3))
        tmp_map.fill(255)
        tmp_map[self.map > 0.7] = [100, 100, 100]
        cv2.imwrite(f'./map/map_{frame_count}.png', tmp_map)'''
        
        info = {}
        info['pos'] = self.frame.avatar_transform.position
        info['rotation'] = self.frame.avatar_transform.rotation
    
    def check_occupied(self, x: float, z: float):
        """
        Check (x, z) if occupied or free or out-side
        0: occupied; 1: free; 2: out_side
        """
        
        if self.occupancy_map is None or self._scene_bounds is None:            
            return False, 0, 0, 0
        i = int(round((x - self._scene_bounds["x_min"]) * SCALE))
        j = int(round((z - self._scene_bounds["z_min"]) * SCALE))
        i = min(i, self.occupancy_map.shape[0] - 1)
        i = max(i, 0)
        j = min(j, self.occupancy_map.shape[1] - 1)
        j = max(j, 0)
        try:
            t = self.occupancy_map[i][j]
        except:
            print('error:', i, j)
        return True, i, j, self.occupancy_map[i][j]
    
    '''def get_occupancy_position(self, i, j):
        """
        Converts the position (i, j) in the occupancy map to (x, z) coordinates.

        :param i: The i coordinate in the occupancy map.
        :param j: The j coordinate in the occupancy map.
        :return: Tuple: True if the position is in the occupancy map; x coordinate; z coordinate.
        """

        if self.occupancy_map is None or self._scene_bounds is None:
            return False, 0, 0,
        x = self._scene_bounds["x_min"] + (i / SCALE)
        z = self._scene_bounds["z_min"] + (j / SCALE)
        return x, z'''
    
    def generate_goal(self):
        while True:
            x, z = np.random.random_sample(), np.random.random_sample()
            x = (self._scene_bounds["x_max"] - self._scene_bounds["x_min"]) * \
                                                        x + self._scene_bounds["x_min"]
            z = (self._scene_bounds["z_max"] - self._scene_bounds["z_min"]) * \
                                                        z + self._scene_bounds["z_min"]            
            rep = self.check_occupied(x, z)
            #sx, _, sz = self.frame.avatar_transform.position
            if rep[3] == 1:# and self.l2_distance((sx, sz), (x, z)) > 4:
                return x, z
        return None
    
    def conv2d(self, map, kernel=3):
        from scipy.signal import convolve2d
        conv = np.ones((kernel, kernel))
        return convolve2d(map, conv, mode='same', boundary='fill')
    
    def find_shortest_path(self, st, goal, map = None):
        if map is None:
            map = self.map
        #map: 0-free, 1-occupied
        st_x, _, st_z = st
        g_x, g_z = goal
        _, st_i, st_j, t = self.check_occupied(st_x, st_z)
        #assert t == 1
        _, g_i, g_j, t = self.check_occupied(g_x, g_z)
        #assert t == 1
        dist_map = np.ones_like(map, dtype=np.float32)
        super_map1 = self.conv2d(map, kernel=5)
        dist_map[super_map1 > 0] = 10
        super_map2 = self.conv2d(map)
        dist_map[super_map2 > 0] = 1000
        dist_map[map > 0] = 100000
        
        #print('min dist:', dist_map.min())
        #print('max dist:', dist_map.max())
        #dist_map
        path = pyastar.astar_path(dist_map, (st_i, st_j),
            (g_i, g_j), allow_diagonal=False)
        return path
    
    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5
    
    def check_goal(self, thresold = 1.0):
        x, _, z = self.frame.avatar_transform.position
        gx, gz = self.goal
        d = self.l2_distance((x, z), (gx, gz))
        return d < thresold
        
    def draw(self, traj, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        for dx in range(1):
            for dz in range(1):
                x = traj[0] + dx
                z = traj[1] + dz
                self.paint[x, z] = value
    
    def dep2map(self):  
        #start = time.time()
        pre_map = np.zeros_like(self.map)
        local_known_map = np.zeros_like(self.map, np.int32)
        try:
            depth = self.frame.get_depth_values()
        except:
            return pre_map
        #camera info
        FOV = 54.43222897365458
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))
        
        #Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)
        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth
        
        pc = np.stack((xx, yy, depth, np.ones((xx.shape[0], xx.shape[1]))))  #3, 256, 256
        pc = pc.reshape(4, -1)
        
        E = self.frame.camera_matrix
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        
        rpc = np.dot(inv_E, pc).reshape(4, W, H)
        #pre_map = np.zeros_like(self.map)
        #print('dep2pc time: ', time.time() - start)
        rpc = rpc.reshape(4, -1)
        X = np.rint((rpc[0, :] - self._scene_bounds["x_min"]) * SCALE)
        X = np.maximum(X, 0)
        X = np.minimum(X, self.map_shape[0] - 1)
        Z = np.rint((rpc[2, :] - self._scene_bounds["z_min"]) * SCALE)
        Z = np.maximum(Z, 0)
        Z = np.minimum(Z, self.map_shape[1] - 1)
        depth = depth.reshape(-1)
        index = np.where((depth < 99) & (rpc[1, :] > 0.1) & (rpc[1, :] < 1.3))
        XX = X[index]
        ZZ = Z[index]
        #print(X.dtype, X.shape)
        XX = XX.astype(np.int32)
        ZZ = ZZ.astype(np.int32)
        pre_map[XX, ZZ] = 1
        
        index = np.where((depth < 99) & (rpc[1, :] < 1.0))
        X = X[index]
        Z = Z[index]
        #print(X.dtype, X.shape)
        X = X.astype(np.int32)
        Z = Z.astype(np.int32)
        local_known_map[X, Z] = 1
        self.known_map = np.maximum(self.known_map, local_known_map)
        '''        
        for i in range(W):
            for j in range(H):
                x, y, z = rpc[0:3, i, j]
                if depth[i, j] < 99 and y > 0.2 and y < 2.8:                    
                    _, id_i, id_j, _ = self.check_occupied(x, z)
                    pre_map[id_i, id_j] = 1
        '''
        #print('dep2pc time: ', time.time() - start)
        return pre_map
    
    def add_third_camera(self):
        #if not self.demo:
        return None
        x, y, z = self.frame.avatar_transform.position
        fx, _, fz = self.frame.avatar_transform.forward
        self.add_overhead_camera({"x": -5.5, "y": 8.0, "z": 1.0}, target_object="a", images="cam")        
    
    def get_object_position(self, object_id):
        return self.frame.object_transforms[object_id].position
    
    def get_object_list(self):
        #seg = self.frame.id_pass
        images = self.frame.get_pil_images()
        seg = np.array(images['id'], np.int32)
        #hash = TDWUtils.color_to_hashable(seg)
        #print(seg.shape)
        W, H, _ = seg.shape
        hash = np.zeros((W, H), np.int32)
        hash[:, :] = (seg[:, :, 0] << 16) + (seg[:, :, 1] << 8) + seg[:, :, 2]
        hash = hash.reshape(-1)
        hash = np.unique(hash)
        #print(hash)
        hash = hash[np.where(hash != 0)]
        #object_id = self.segmentation_color_to_id[hash]
        from operator import itemgetter
        self.object_list = {0: [], 1: [], 2: []}
        #if hash[0].shape[0] == 0:
        if len(hash) == 0:
            return
        '''try:
            self.object_ids = itemgetter(*hash)(self.segmentation_color_to_id)
        except:'''
        self.object_ids = []
        for i in hash:
            if i in self.segmentation_color_to_id:
                self.object_ids.append(self.segmentation_color_to_id[i])
        self.object_ids = tuple(self.object_ids)
        #print(type(self.object_ids))
        #self.object_ids = list(self.object_ids)
        if not isinstance(self.object_ids, tuple):
            #print(type(self.object_ids), self.object_ids)
            #print(self.static_object_info[self.object_ids].model_name)
            self.object_ids = [self.object_ids]
        #print(self.object_id)
                
        for id in self.object_ids:
            if id not in self.static_object_info:
                print('not in static:', id)
                continue
            if self.static_object_info[id].target_object:
                if id in self.finish:
                    continue
                x, y, z = self.frame.object_transforms[id].position
                _, i, j, _ = self.check_occupied(x, z)
                self.explored_map[i, j] = 1
                self.id_map[i, j] = id
                self.object_list[0].append(id)
            elif self.static_object_info[id].container:
                x, y, z = self.frame.object_transforms[id].position
                _, i, j, _ = self.check_occupied(x, z)
                self.explored_map[i, j] = 2
                self.id_map[i, j] = id
                self.object_list[1].append(id)
            elif self.static_object_info[id].model_name == self.goal_object:
                x, y, z = self.frame.object_transforms[id].position
                _, i, j, _ = self.check_occupied(x, z)
                self.explored_map[i, j] = 3
                self.id_map[i, j] = id
                self.object_list[2].append(id)
            
    def grasp(self, object_id, arm):
        object_id = int(object_id)
        self.turn_to(object_id)
        if arm == Arm.left:
            d_theta = 15
        else:   
            d_theta = -15
        grasped = False
        theta = 0
        while theta <= 90 and not grasped:
            # Try to grasp the object.
            start = time.time()
            status = self.grasp_object(object_id=object_id,
                                    arm=arm,
                                    check_if_possible=False,
                                    stop_on_mitten_collision=True)
            print('grasp time:', time.time() - start)
            print(status)
            if status == TaskStatus.success and \
                    object_id in self.frame.held_objects[arm]:
                grasped = True
                break
            # Turn a bit and try again.
            if not grasped:
                self.turn_by(d_theta)
                theta += abs(d_theta)
            print('theta:', theta)
        return grasped
    
    
    
    def move_to(self, max_move_step = 100, d = 0.7):
        move_step = 0
        self.traj = []
        self.draw_Astar_map()
        print(self.map_num)
        while not self.check_goal(d) and move_step < max_move_step:
            step_time = time.time()
            self.step += 1
            move_step += 1
            self.position = self.frame.avatar_transform.position
            
            self.get_object_list()
            pre_map = self.dep2map()
            self.map = np.maximum(self.map, pre_map)
            
            path = self.find_shortest_path(self.position, self.goal, self.gt_map)
            i, j = path[min(2, len(path) - 1)]
            x, z = self.get_occupancy_position(i, j)
            #assert T == True
            local_goal = [x, z]
            angle = TDWUtils.get_angle(forward=np.array(self.frame.avatar_transform.forward),
                                origin=np.array(self.frame.avatar_transform.position),
                                position=np.array([local_goal[0], 0, local_goal[1]]))
            #print('angle:', angle)
            action_time = time.time()
            if np.abs(angle) < ANGLE:
                px, py, pz = self.frame.avatar_transform.position
                status = self.move_forward_by(distance=DISTANCE,
                                                move_force = 300,
                                                move_stopping_threshold=0.12,
                                                num_attempts = 30)                
                x, y, z = self.frame.avatar_transform.position
                _, i, j, _ = self.check_occupied(x, z)
                self.traj.append((i, j))
                action = 0
                if self.l2_distance((px, pz), (x, z)) < 0.1:
                    x, y, z = np.array(self._avatar.frame.get_position()) + (np.array(self._avatar.frame.get_forward()) * 0.25)
                    _, i, j, _ = self.check_occupied(x, z)
                    self.map[i, j] = 1
                #if status == TaskStatus.too_long or status == TaskStatus.collided_with_environment:
                #    x, y, z = np.array(self._avatar.frame.get_position()) + (np.array(self._avatar.frame.get_forward()) * 0.25)
                #    _, i, j, _ = self.check_occupied(x, z)
                #    self.gt_map[i, j] = 1
            elif angle > 0:
                status = self.turn_by(angle=-ANGLE, force=1200, num_attempts=25)
                action = 1
            else:
                status = self.turn_by(angle=ANGLE, force=1200, num_attempts=25)
                action = 2
            
            self.action_list.append(action)    
            action_time = time.time() - action_time
            step_time = time.time() - step_time
            x, y, z = self.frame.avatar_transform.position            
            self.f.write('step: {}, position: {}, action: {}, goal: {}\n'.format(
                    self.step, \
                    self.frame.avatar_transform.position, \
                    action, \
                    self.goal                    
                ))
            self.f.write('local_goal: {}, distance: {}, angle: {}, forward: {}\n'.format(
                    local_goal, \
                    self.l2_distance((x, z), self.goal), \
                    angle, \
                    self.frame.avatar_transform.forward
                ))
            self.f.write('status: {}, action time: {}, step time: {}\n'.format(
                status, action_time, step_time))
            
            self.f.flush()
        
        self.draw_map()
        return self.check_goal()
    
    def interact(self, object_id):
        object_id = int(object_id)
        self.frame.save_images(self.output_dir + f'/self.sub_goal')
        print('??')
        #print(self.static_object_info[object_id].model_name)
        #d = np.linalg.norm(self.frame.avatar_transform.position - self.frame.object_transforms[object_id].position)
        #self.step += d / 0.5        
        if self.sub_goal == 2:
            lift_high = True
        else:
            lift_high = False
        holding_arms = []
        for a in self.frame.held_objects:
            if len(self.frame.held_objects[a]) > 0:
                holding_arms.append(a)
        for a in holding_arms:
            self._lift_arm(arm=a, lift_high=lift_high)
        print('!!')
        Flag = False
        
        x, y, z = self.frame.object_transforms[object_id].position
        self.goal = (x, z)
        if not self.move_to():
            return False
         
        d = np.linalg.norm(self.frame.avatar_transform.position - self.frame.object_transforms[object_id].position)
        print('distance:', d)
        print(object_id)
        print(self.static_object_info[object_id].model_name)
        #self.turn_to(object_id)
        if self.sub_goal < 2:
            print(self.go_to(object_id, move_stopping_threshold=0.3))
        else:
            print(self.go_to(object_id, move_stopping_threshold=0.7))
        for i in range(3):
            d = np.linalg.norm(self.frame.avatar_transform.position - self.frame.object_transforms[object_id].position)
            print('distance:', d)
            if d > 0.7:
                for a in holding_arms:
                    self.reset_arm(arm=a)
                    self._lift_arm(arm=a, lift_high=lift_high)
                self.go_to(object_id, move_stopping_threshold=0.3)
            else:
                Flag = True
                break
        if d > 1.0:
            return False
        for a in holding_arms:
            self._lift_arm(arm=a, lift_high=lift_high)
        
        if self.sub_goal < 2:
            grasp_arms = []
            
            for a in self.frame.held_objects:
                if len(self.frame.held_objects[a]) == 0:
                    grasp_arms.append(a)
            Flag = False
            arm = None
            for a in grasp_arms:
                Flag = self.grasp(object_id, a)
                if Flag:
                    arm = a
                    break
            if Flag and self.static_object_info[object_id].container:
                self.content_container[object_id] = []
            if Flag and self.container_held is not None:
                print('???')
                start = time.time()
                print(self.frame.held_objects[Arm.left], self.frame.held_objects[Arm.right])
                status = self.put_in_container(object_id=object_id,
                                                container_id = self.container_held,
                                                arm=arm)
                print('put_in_container:', status, time.time() - start)
                print(self.frame.held_objects[Arm.left], self.frame.held_objects[Arm.right])                
                print('!!!')
                if status != TaskStatus.success:
                    if arm == Arm.left:
                        arm = Arm.right
                    else:
                        arm = Arm.left
                    status = self.grasp(self.container_held, arm = arm)
                self.content_container[self.container_held].append(object_id)
                if self.static_object_info[object_id].target_object:
                    self.update_held(object_id)
        else:
            Flag = True            
            for id in self.held_objects:
                if self.is_container(id):
                    self.finish[id] = 1
                    content = self.content_container[id]
                    #overlap_ids = self._get_objects_in_container(container_id=id)
                    for o in content:
                        self.finish[o] = 1
                        self.update_finish(o)
                    self.content_container[id] = []
                else:
                    self.finish[id] = 1
                    self.update_finish(id)
            for a in holding_arms:
                self.drop(a)
        return Flag
    
    def nav(self, max_nav_step):
        action = 0
        nav_step = 0
        while not self.check_goal() and nav_step < max_nav_step:
            step_time = time.time()
            self.step += 1
            nav_step += 1
            self.position = self.frame.avatar_transform.position
            
            self.get_object_list()
            pre_map = self.dep2map()
            self.map = np.maximum(self.map, pre_map)
            
            if len(self.object_list[1]) > 0:
                if self.sub_goal < 2 and not self.container_held and \
                                self.target_object_held.sum() > 2:
                    print('begin interact with container')
                    self.sub_goal = 1
                    goal = random.choice(self.object_list[1])
                    self.interact(goal)
                    return
            
            if len(self.object_list[self.sub_goal]) > 0:
                #interact
                print('begin interact')
                goal = random.choice(self.object_list[self.sub_goal])
                self.interact(goal)
                return
            
            path = self.find_shortest_path(self.position, self.goal, self.gt_map)
            i, j = path[min(2, len(path) - 1)]
            x, z = self.get_occupancy_position(i, j)
            #assert T == True
            local_goal = [x, z]
            angle = TDWUtils.get_angle(forward=np.array(self.frame.avatar_transform.forward),
                                origin=np.array(self.frame.avatar_transform.position),
                                position=np.array([local_goal[0], 0, local_goal[1]]))
            #print('angle:', angle)
            action_time = time.time()
            if np.abs(angle) < ANGLE:
                px, py, pz = self.frame.avatar_transform.position
                status = self.move_forward_by(distance=DISTANCE,
                                                move_force = 300,
                                                move_stopping_threshold=0.12,
                                                num_attempts = 30)                
                x, y, z = self.frame.avatar_transform.position
                _, i, j, _ = self.check_occupied(x, z)
                self.traj.append((i, j))
                action = 0
                if self.l2_distance((px, pz), (x, z)) < 0.1:
                    x, y, z = np.array(self._avatar.frame.get_position()) + (np.array(self._avatar.frame.get_forward()) * 0.25)
                    _, i, j, _ = self.check_occupied(x, z)
                    self.map[i, j] = 1
                #if status == TaskStatus.too_long or status == TaskStatus.collided_with_environment:
                #    x, y, z = np.array(self._avatar.frame.get_position()) + (np.array(self._avatar.frame.get_forward()) * 0.25)
                #    _, i, j, _ = self.check_occupied(x, z)
                #    self.gt_map[i, j] = 1
            elif angle > 0:
                status = self.turn_by(angle=-ANGLE, force=1200, num_attempts=25)
                action = 1
            else:
                status = self.turn_by(angle=ANGLE, force=1200, num_attempts=25)
                action = 2
            
            self.action_list.append(action)    
            action_time = time.time() - action_time
            step_time = time.time() - step_time
            x, y, z = self.frame.avatar_transform.position            
            self.f.write('step: {}, position: {}, action: {}, goal: {}\n'.format(
                    self.step, \
                    self.frame.avatar_transform.position, \
                    action, \
                    self.goal                    
                ))
            self.f.write('local_goal: {}, distance: {}, angle: {}, forward: {}\n'.format(
                    local_goal, \
                    self.l2_distance((x, z), self.goal), \
                    angle, \
                    self.frame.avatar_transform.forward
                ))
            self.f.write('status: {}, action time: {}, step time: {}\n'.format(
                status, action_time, step_time))
            
            self.f.flush()
    
    def _lift_arm(self, arm: Arm, lift_high=False) -> None:
        """
        Lift the arm up.

        :param arm: The arm.
        """
        start = time.time()
        if lift_high:
            y = 0.6
        else:
            y = 0.4
        status = self.reach_for_target(arm=arm,
                              target={"x": -0.2 if arm == Arm.left else 0.2, "y": y, "z": 0.3},
                              check_if_possible=True,
                              stop_on_mitten_collision=True)
        print('lift time:', status, time.time() - start)
            
    def draw_map(self):
        self.map_num += 1
        if len(self.traj) == 0:
            return
        self.map_num += 1
        W, H = self.map.shape
        self.paint = np.zeros((W, H, 3))
        self.paint.fill(100)
        self.paint[self.map == 0, 0:3] = 255
        k_f = max(255 / len(self.traj), 2.5)
        for i in range(len(self.traj)):
            self.draw(self.traj[i], [i * k_f, 255, 255 - i * k_f])
        _, i, j, _ = self.check_occupied(self.goal[0], self.goal[1])
        self.draw((i, j), [255, 0, 255])
        
        '''print('len:', len(self.static_object_info))
        for i in range(self.n):
            x, y, z = self.get_object_position(self.goal_idx[i])
            _, i, j, _ = self.check_occupied(x, z)
            self.draw((i, j), [120, 0, 120])'''
        H, W, _ = self.paint.shape
        self.paint = cv2.resize(self.paint, dsize=(0, 0), fx = 4, fy = 4)
        cv2.imwrite(f'./{self.output_dir}/map{self.map_num}.jpg', self.paint)   
    
    
    def draw_Astar_map(self):
        W, H = self.map.shape
        self.paint = np.zeros((W, H, 3))
        self.paint.fill(100)
        self.paint[self.gt_map == 0, 0:3] = 255
        
        self.position = self.frame.avatar_transform.position
        path = self.find_shortest_path(self.position, self.goal, self.gt_map)
        
        k_f = max(255 / len(path), 2.5)
        for t in range(len(path)):
            self.draw(path[t], [t * k_f, 255, 255 - t * k_f])            
        _, i, j, _ = self.check_occupied(self.goal[0], self.goal[1])
        self.draw((i, j), [255, 0, 255])
        H, W, _ = self.paint.shape
        self.paint = cv2.resize(self.paint, dsize=(0, 0), fx = 4, fy = 4)
        cv2.imwrite(f'./{self.output_dir}/A_map{self.map_num}.jpg', self.paint)   
        
    
    def is_container(self, id):
        return self.static_object_info[id].container
    
    def container_full(self, container):
        start = time.time()
        overlap_ids = self._get_objects_in_container(container_id=container)        
        end = time.time()
        if end - start > 0.1:
            print('full takes too much time')
        return len(overlap_ids) > 3
    
    def decide_sub(self):
        self.held_objects = []
        self.held_objects.extend(self.frame.held_objects[Arm.left])
        self.held_objects.extend(self.frame.held_objects[Arm.right])
        if self.target_object_held.sum() == 0:
            #all objects are found
            self.sub_goal = 2
        #elif self.step > self.max_steps - 50 and len(self.held_objects) > 0
        else:
            self.container_held = None
            if len(self.held_objects) > 0:
                for o in self.held_objects:
                    if self.is_container(o):
                        self.container_held = o
                        if self.container_full(o):
                            self.sub_goal = 2
                        else:
                            self.sub_goal = 0
                        return
            if self.container_held is None:
                #self.find_container_step < self.max_container_step and \
                if self.target_object_held.sum() > 2:
                    self.sub_goal = 1
                else:
                    self.sub_goal = 0             
    
    def ex_goal(self):
        while True:
            goal = np.where(self.known_map == 0)
            idx = random.randint(0, goal[0].shape[0] - 1)
            i, j = goal[0][idx], goal[1][idx]
            if self.gt_map[i, j] == 0:
                self.goal = self.get_occupancy_position(i, j)        
                break
    
    def update_finish(self, id):
        name = self.static_object_info[id].model_name
        print('final finish:', name)
        self.target_object_list[self.target_object_dict[name]] -= 1 
        
    def update_held(self, id):
        name = self.static_object_info[id].model_name
        print('held finish:', name)
        self.target_object_held[self.target_object_dict[name]] -= 1 
    
    
    def draw_init_map(self, num):
        self.surface_object_categories = \
                    loads(SURFACE_OBJECT_CATEGORIES_PATH.read_text(encoding="utf-8"))
        W, H = self.occupancy_map.shape
        self.paint = np.zeros((W, H, 3))
        self.paint.fill(100)
        self.paint[self.occupancy_map == 1, 0:3] = 255
        for id in self.static_object_info:
            if self.static_object_info[id].container:
                print('container:', self.static_object_info[id].model_name)
                x, y, z = self.frame.object_transforms[id].position
                _, i, j, _ = self.check_occupied(x, z)
                self.paint[i, j] = np.array([220,20,60])    #red
            name = self.static_object_info[id].model_name
            if name in self.surface_object_categories and \
                self.surface_object_categories[name] == self.goal_object:
                #print(name, self.goal_object)
                x, y, z = self.frame.object_transforms[id].position
                _, i, j, _ = self.check_occupied(x, z)
                self.paint[i, j] = np.array([0,255,0])    #green
        for id in self._target_object_ids:
            x, y, z = self.frame.object_transforms[id].position
            _, i, j, _ = self.check_occupied(x, z)
            self.paint[i, j] = np.array([0, 0, 255])    #blue
        
        self.paint = cv2.resize(self.paint, dsize=(0, 0), fx = 4, fy = 4)
        cv2.imwrite(f'./map{num}.jpg', self.paint)   
        
    def run(self, scene='2a', layout=1, output_dir='transport') -> None:
        """
        Run a single trial. Save images per frame.
        """        
        if isinstance(output_dir, str):
            output_path = Path(output_dir)
        if not output_path.exists():
            output_path.mkdir(parents=True)

        self.output_dir = output_dir
        output_directory = output_dir        
        start_time = time.time()
        self.init_scene(scene=scene, layout=layout, room = 0)        
        print('init scene time:', time.time() - start_time)
        self.action_list = []
        for i in range(5):
            self.map_shape = self.occupancy_map.shape
            self.draw_init_map(i)
            self.init_scene(scene=scene, layout=layout, room = 0)
            
        return
        if DEMO:
            self.add_overhead_camera({"x": 5.8, "y": 4.0, "z": -1.3}, target_object="a", images="cam")        
    
        #return None
        #map: 0: free, 1: occupied
        #occupancy map: 0: occupied; 1: free; 2: out-side
        self.gt_map = np.zeros_like(self.occupancy_map)
        self.gt_map[self.occupancy_map != 1] = 1
        
        self.map = np.zeros_like(self.occupancy_map)
        #0: unknown, 1: known
        self.known_map = np.zeros_like(self.occupancy_map, np.int32)
        #0: unknown, 1: target object, 2: container, 3: goal, 4: free
        self.explored_map = np.zeros_like(self.occupancy_map, np.int32)
        #0: unknown, 1: object_id(only target and container)
        self.id_map = np.zeros_like(self.occupancy_map, np.int32)
        
        print('occupancy_map shape:', self.occupancy_map.shape)
        W, H = self.occupancy_map.shape
        self.map_shape = self.occupancy_map.shape        
        
        x, y, z = self.frame.avatar_transform.position
        self.position = self.frame.avatar_transform.position
        print('init position:', x, y, z)
        #assert self.check_occupied(x, z)[3] == 1
        W, H = self.occupancy_map.shape
        
        self.step = 0
        self.map_num = 0
        action = 0
        
        self.goal_idx = []
        #self._target_object_ids: list
        self.container = []
        for i in self.static_object_info:
            if self.static_object_info[i].container:
                self.container.append(i)
        
        self.target_object_dict = {'vase_02': 0, 'jug04': 1, 'jug05': 2,
                                'elephant_bowl': 0,
                                'hexagonal_toy': 0}
        self.target_object_list = np.zeros(3, np.int32)
        self.target_object_held = np.zeros(3, np.int32)
        for i in self._target_object_ids:
            name = self.static_object_info[i].model_name
            #finish
            self.target_object_list[self.target_object_dict[name]] += 1 
            #find but still held
            self.target_object_held[self.target_object_dict[name]] += 1
        self.total_target_object = self.target_object_list.sum()    
        self.goal_dict = {'bed': 0,
                            'table': 1,
                            'coffee table': 2,
                            'sofa': 3,
                            'bench': 4}
        #self.goal_vec = np.zeros(5, np.int32)
        #self.goal_vec[self.goal_dict[self.goal_object]] += 1
        self.goal_object = "coffee table"
        #self.goal_idx.append(random.choice(self._target_object_ids))
        #self.goal_idx.append(random.choice(self.container))
        self.finish = {}
        self.content_container = {}
        
        self.f = open(f'./{output_dir}/nav.log', 'w')
        ff = open(f'./{output_dir}/object.log', 'w')
        status = ''
        self.traj = []
        
        self.step = 0
        #0: find objects
        #1: find container
        #2: transport to goal
        #3: Nav
        self.sub_goal = 1
        self.nav_goal = 0
        tot_start = time.time()
        while self.step < self.max_steps and self.target_object_list.sum() > 0:
            #make sure sub_goal
            print(self.target_object_held.sum(), self.target_object_list.sum())
            self.decide_sub()
            
            goal = np.where(self.explored_map == self.sub_goal + 1)
            if goal[0].shape[0] > 0:                
                idx = random.randint(0, goal[0].shape[0] - 1)
                i, j = goal[0][idx], goal[1][idx]
                self.goal = self.get_occupancy_position(i, j)
                self.interact(self.id_map[i, j])
            else:
                self.ex_goal()
                self.nav(self.step + 30)
        self.decide_sub()
        print(self.held_objects)
        
        print('fianl time:', time.time()  - tot_start)
        
        '''for i in range(2):
            fff.write(f' {self.static_object_info[self.object_id].model_name} ')
            self.draw_Astar_map()
            self.nav()
            self.draw_map()
            fff.write(f'goal{i}_{self.check_goal()} ')
            return
            #assert self.check_goal()
            #if self.check_goal():
            print('check_goal:', self.check_goal())
            if True:
                self.step = 0
                
                self.action_list.append(4)
                if self.idx > 0:
                    print('turn to: ', self.turn_to(target=self.object_id, force=300))
                    self.arm = Arm.right
                    self.reset_arm(Arm.right)
                    self.lift_arm(Arm.right)
                else:
                    self.arm = Arm.left
                print('go to: ', self.go_to(target=self.object_id, move_stopping_threshold=0.3))
                start = time.time()
                grasped = self.grasp(self.object_id, self.arm)
                print('grasp:', grasped, self.frame.held_objects[self.arm])             
                print('grasp time:', time.time() - start)
                print(self.frame.held_objects[self.arm])
                fff.write(f'{i}_{grasped} ')
                if self.idx > 0:
                    print(self.reset_arm(arm=Arm.right))
                    print(self.reset_arm(arm=Arm.left))                    
                    self.lift_arm(arm=Arm.right)
                    s = len(self.frame.held_objects[Arm.left]) > 0 and len(self.frame.held_objects[Arm.right]) > 0
                    print(s)
                    fff.write(f'both_{s} ')
                    status = self.put_in_container(object_id=self.goal_idx[0],
                                            container_id=self.goal_idx[1],
                                            arm=Arm.left)
                    print('hold:', status)
                    if status == TaskStatus.success:
                        fff.write('h_True ')
                        status = self.pour_out_container(arm=Arm.right)
                        print('pour out:', status)
                        if status == TaskStatus.success:
                            fff.write('p_True ')
                        else:
                            fff.write('p_False ')
                    else:
                        fff.write('h_False ')
                    
                    self.reset_arm(arm=Arm.right)
                    self.reset_arm(arm=Arm.left)                    
                    self.lift_arm(arm=Arm.right)
                    
                
                ff.write('object_id: {}, position: {}, name: {}, grasp: {}\n'.format(
                        self.object_id,
                        self.get_object_position(self.object_id),
                        self.static_object_info[self.object_id].model_name,
                        grasped))
                self.idx += 1
                if self.idx == self.n:
                    break
                self.object_id = self.goal_idx[self.idx]
                x, y, z = self.get_object_position(self.object_id)
                self.goal = (x, z)
                
                print('goal position:', self.goal)'''
            
        
        
                
        with open('action.txt', 'wb') as fa:
            pickle.dump(self.action_list, fa)
        self.f.flush()
        ff.flush()
        #self.end()


if __name__ == "__main__":
    port = 1076
    docker_id = create_tdw(port=port)
    c = Nav(port=port, launch_build=False, demo=False)
    try:        
        for i in range(1):
            print('epoch ', i)
            fff.write(f'\nepoch {i}:')
            c.run(output_dir='trans0')
            fff.flush()
    finally:        
        kill_tdw(docker_id)