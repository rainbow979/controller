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
import time
import cv2
import math
import pickle
from sticky_mitten_avatar.task_status import TaskStatus
from sticky_mitten_avatar.paths import SURFACE_OBJECT_CATEGORIES_PATH
import argparse

import random

DISTANCE = 0.5
ANGLE = 15.

DEMO = False

content_container = {}
total_finish = 0
total_grasp = 0

def update(c):
    global total_finish
    c.held_objects = []
    c.held_objects.extend(c.frame.held_objects[Arm.left])
    c.held_objects.extend(c.frame.held_objects[Arm.right])            
    for id in c.held_objects:
        if c.static_object_info[id].container:            
            content = content_container[id]#c.content_container[id]
            #overlap_ids = self._get_objects_in_container(container_id=id)
            for o in content:
                total_finish += 1
            content_container[id] = []
        else:
            total_finish += 1

def my_put_in_container(c, object_id, container_id, arm):
    #c.action_list.append([5, object_id, container_id, arm == Arm.right])
    #print('before position:', self.frame.object_transforms[object_id].position)
    #try:
    content_container[container_id].append(object_id)
    c._start_task()
    c.drop(arm)        
    position = {'x': float(20 + random.randint(0, 10)), 
                'y': float(0.),
                'z': float(20 + random.randint(0, 10))}
    
    c.communicate([{"$type": "teleport_object",
           "position": position,
           "id": object_id,
           "physics": True}])
    c._end_task() 
    #except:
    #    print('put_in_container Error:', container_id)
    #print('after position:', self.frame.object_transforms[object_id].position)
    return True


if __name__ == "__main__":
    port = 18015
    docker_id = create_tdw(port=port)
    c = StickyMittenAvatarController(port=port,
                                    launch_build=False,
                                    id_pass = True, 
                                    demo=DEMO,
                                    train=2)
    data_id = 44
    arm = None
    
    try:
        c.init_scene(data_id = data_id)        
        if DEMO:
            c.communicate([{"$type": "set_floorplan_roof", "show": False}])
            c.add_overhead_camera({"x": 0, "y": 8.0, "z": 0},
                                    target_object="a",
                                    images="cam")      
        #c.frame.save_images(output_directory='./demo')
        #assert False
        with open('action.pkl', 'rb') as f:
            action_list = pickle.load(f)
        for l in action_list:
            if l[0] == 0:
                c.move_forward_by(distance=DISTANCE,
                                move_force = 300,
                                move_stopping_threshold=0.2,
                                num_attempts = 25)
            elif l[0] == 1:
                c.turn_by(angle=-ANGLE, force=1000, num_attempts=25)                
            elif l[0] == 2:
                c.turn_by(angle=ANGLE, force=1000, num_attempts=25)
            elif l[0] == 3:
                #print(l[1])
                object_id = c.demo_id_to_object[l[1]]
                c.go_to(object_id, move_stopping_threshold=l[2])
            elif l[0] == 4:
                if l[2] == 0:
                    arm = Arm.left
                else:
                    arm = Arm.right
                object_id = c.demo_id_to_object[l[1]]
                print('grasp:', object_id)
                status = c.grasp_object(object_id=object_id,
                                    arm=arm,
                                    check_if_possible=True,
                                    stop_on_mitten_collision=True)
                if status == TaskStatus.success:
                    print('grasp success')
                    if c.static_object_info[object_id].target_object:
                        total_grasp += 1
                    else:
                        if object_id not in content_container:
                            content_container[object_id] = []
            elif l[0] == 5:
                if l[3] == 0:
                    arm = Arm.left
                else:
                    arm = Arm.right
                object_id = c.demo_id_to_object[l[1]]
                print(l[2])
                container_id = c.demo_id_to_object[l[2]]
                my_put_in_container(c, object_id, container_id, arm)
            elif l[0] == 6:
                update(c)
                if len(c.frame.held_objects[Arm.left]) > 0:
                    c.drop(Arm.left)
                if len(c.frame.held_objects[Arm.right]) > 0:
                    c.drop(Arm.right)
    finally:
        kill_tdw(docker_id)
        print(total_grasp, total_finish)