from csv import DictReader
from json import loads
from pathlib import Path
import random
import numpy as np
from pkg_resources import resource_filename
from typing import Dict, List, Union, Optional, Tuple
from tdw.floorplan_controller import FloorplanController
from tdw.tdw_utils import TDWUtils, QuaternionUtils
from tdw.output_data import Bounds, Rigidbodies, SegmentationColors, Raycast, CompositeObjects, Overlap, Transforms,\
    Version
from tdw.py_impact import AudioMaterial, PyImpact, ObjectInfo
from tdw.object_init_data import AudioInitData
from tdw.release.pypi import PyPi
from sticky_mitten_avatar.avatars import Arm, Baby
from sticky_mitten_avatar.avatars.avatar import Avatar, BodyPartStatic
from sticky_mitten_avatar.util import get_data, OCCUPANCY_CELL_SIZE, \
    TARGET_OBJECT_MASS, CONTAINER_MASS, CONTAINER_SCALE
from sticky_mitten_avatar.paths import SPAWN_POSITIONS_PATH, OCCUPANCY_MAP_DIRECTORY, SCENE_BOUNDS_PATH, \
    ROOM_MAP_DIRECTORY, Y_MAP_DIRECTORY, TARGET_OBJECTS_PATH, COMPOSITE_OBJECT_AUDIO_PATH, SURFACE_MAP_DIRECTORY, \
    TARGET_OBJECT_MATERIALS_PATH, OBJECT_SPAWN_MAP_DIRECTORY
from sticky_mitten_avatar.static_object_info import StaticObjectInfo
from sticky_mitten_avatar.frame_data import FrameData
from sticky_mitten_avatar.task_status import TaskStatus

import pickle

def get_occupancy_position(i, j, _scene_bounds):
    x = _scene_bounds["x_min"] + (i * OCCUPANCY_CELL_SIZE)
    z = _scene_bounds["z_min"] + (j * OCCUPANCY_CELL_SIZE)
    return x, z

def generate_data(scene, layout, room):

    data = {}
    data['scene'] = {'scene': scene, 'layout': layout, 'room': room}
    

    _scene_bounds = loads(SCENE_BOUNDS_PATH.read_text())[scene[0]]
    room_map = np.load(str(ROOM_MAP_DIRECTORY.joinpath(f"{scene[0]}.npy").resolve()))
    map_filename = f"{scene[0]}_{layout}.npy"
    occupancy_map = np.load(
        str(OCCUPANCY_MAP_DIRECTORY.joinpath(map_filename).resolve()))
    map = np.zeros_like(occupancy_map)
    map[occupancy_map != 1] = 1
    from scipy.signal import convolve2d
    conv = np.ones((5, 5))
    map = convolve2d(map, conv, mode='same', boundary='fill')
    
    ys_map = np.load(str(Y_MAP_DIRECTORY.joinpath(map_filename).resolve()))
    object_spawn_map = np.load(str(OBJECT_SPAWN_MAP_DIRECTORY.joinpath(map_filename).resolve()))
    
    all_placeable_positions = []
    rooms: Dict[int, List[Tuple[int, int]]] = dict()
    for i in range(0, np.amax(room_map)):
        # Get all free positions in the room.
        placeable_positions: List[Tuple[int, int]] = list()
        for ix, iy in np.ndindex(room_map.shape):
            if room_map[ix][iy] == i:
                # If this is a spawnable position, add it.
                if object_spawn_map[ix][iy] and map[ix, iy] == 0:
                    placeable_positions.append((ix, iy))
        if len(placeable_positions) > 0:
            rooms[i] = placeable_positions
            all_placeable_positions.extend(placeable_positions)
    data['container'] = []
    #print(rooms.keys())
    num_container = 0
    rooms_list = list(rooms.keys())
    random.shuffle(rooms_list)
    for room_key in rooms_list:
        # Maybe don't add a container in this room.        
        if random.random() < 0.25:
            continue
        if num_container == 4:
            continue
        num_container += 1
        proc_gen_positions = rooms[room_key][:]
        random.shuffle(proc_gen_positions)
        # Get a random position in the room.
        ix, iy = random.choice(rooms[room_key])

        # Get the (x, z) coordinates for this position.
        # The y coordinate is in `ys_map`.
        x, z = get_occupancy_position(ix, iy, _scene_bounds)
        container_name = "basket_18inx18inx12iin"
        position={"x": x, "y": ys_map[ix][iy], "z": z}
        rotation={"x": 0, "y": random.uniform(-179, 179), "z": 0}
        scale=CONTAINER_SCALE
        data['container'].append({
            'name': container_name,
            'ixy': (ix, iy),
            'position': position,
            'rotation': rotation,
            'scale': scale
        })
        '''container_id, container_commands = self._add_object(position={"x": x, "y": ys_map[ix][iy], "z": z},
                                                            rotation={"x": 0,
                                                                      "y": random.uniform(-179, 179),
                                                                      "z": z},
                                                            scale=CONTAINER_SCALE,
                                                            audio=self._default_audio_values[
                                                                container_name],
                                                            model_name=container_name)'''
    data['target_object'] = []    
    target_objects: Dict[str, float] = dict()
    with open(str(TARGET_OBJECTS_PATH.resolve())) as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            target_objects[row["name"]] = float(row["scale"])
    target_object_names = list(target_objects.keys())
    target_object_names = target_object_names[:3]
    #print(target_object_names)
    # Load a list of visual materials for target objects.
    target_object_materials = TARGET_OBJECT_MATERIALS_PATH.read_text(encoding="utf-8").split("\n")

    # Get all positions in the room and shuffle the order.
    #target_room_positions = random.choice(list(rooms.values()))
    #print(len(target_room_positions))
    #print(len(all_placeable_positions))
    target_room_positions = all_placeable_positions
    random.shuffle(target_room_positions)
    # Add the objects.
    _target_object_ids = []
    for i in range(random.randint(8, 12)):
        ix, iy = random.choice(target_room_positions)
        # Get the (x, z) coordinates for this position.
        # The y coordinate is in `ys_map`.
        x, z = get_occupancy_position(ix, iy, _scene_bounds)
        target_object_name = random.choice(target_object_names)
        # Set custom object info for the target objects.
        #audio = ObjectInfo(name=target_object_name, mass=TARGET_OBJECT_MASS, material=AudioMaterial.ceramic,
        #                   resonance=0.6, amp=0.01, library="models_core.json", bounciness=0.5)
        scale = target_objects[target_object_name]
        position={"x": x, "y": ys_map[ix][iy], "z": z}
        rotation={"x": 0, "y": random.uniform(-179, 179), "z": 0}
        visual_material = random.choice(target_object_materials)
        data['target_object'].append({
            'name': target_object_name,
            'position': position,
            'rotation': rotation,
            'scale': scale,
            'visual_material': visual_material,
            'ixy': (ix, iy)
        })
        '''object_id, object_commands = self._add_object(position={"x": x, "y": ys_map[ix][iy], "z": z},
                                                      rotation={"x": 0, "y": random.uniform(-179, 179),
                                                                "z": z},
                                                      scale={"x": scale, "y": scale, "z": scale},
                                                      audio=audio,
                                                      model_name=target_object_name)'''
        #_target_object_ids.append(object_id)            

        # Set a random visual material for each target object.
        
    _goal_positions = loads(SURFACE_MAP_DIRECTORY.joinpath(f"{scene[0]}_{layout}.json").
                                   read_text(encoding="utf-8"))
    goal_positions = dict()
    goal_objects = []
    for k, v in _goal_positions.items():        
        goal_positions[int(k)] = _goal_positions[k]
        for k1 in v:
            goal_objects.append(k1)
    goal_objects = list(set(goal_objects))
    goal_object = random.choice(goal_objects)
    data['goal_object'] = goal_object
    
    rooms = loads(SPAWN_POSITIONS_PATH.read_text())[scene[0]][str(layout)]
    if room == -1:
        room = random.randint(0, len(rooms) - 1)
    assert 0 <= room < len(rooms), f"Invalid room: {room}"
    data['scene']['room'] = room
    avatar_position = rooms[room]
    
    data['avatar_position'] = avatar_position
    return data

if __name__ == '__main__':
    import time
    start = time.time()
    scene = '1a'
    layout = 0
    room = -1
    dataset = []
    train = False
    if train:
        scenes = ['2a', '2b', '5a', '5b', '1a']
        layouts = [0, 1]
        l = 100
        path = 'train_dataset.pkl'
    else:   
        scenes = ['2a', '2b', '5a', '5b']
        layouts = [2]        
        l = 25
        path = 'test_dataset.pkl'
    for scene in scenes:
        for layout in layouts:
            for i in range(l):                
                dataset.append(generate_data(scene, layout, room))
    print(len(dataset))
    random.shuffle(dataset)
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print(time.time() - start)