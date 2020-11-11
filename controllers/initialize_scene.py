from sticky_mitten_avatar import StickyMittenAvatarController
from utils import create_tdw, kill_tdw
  
from argparse import ArgumentParser
from json import loads
from pathlib import Path
import numpy as np
from tdw.tdw_utils import TDWUtils
from tdw.output_data import Images
from tdw.floorplan_controller import FloorplanController
from sticky_mitten_avatar.util import get_data, OCCUPANCY_CELL_SIZE
from sticky_mitten_avatar.paths import OCCUPANCY_MAP_DIRECTORY, SCENE_BOUNDS_PATH, OBJECT_SPAWN_MAP_DIRECTORY
"""
This is a simple example of how to initialize an interior environment populated by furniture, objects, and an avatar.
"""

if __name__ == "__main__":
    port = 10075
    docker_id = create_tdw(port=port)
    # Instantiate the controller.
    try:
        c = StickyMittenAvatarController(port = port, launch_build=False)
        # Initialize the scene. Populate it with objects. Spawn the avatar in a room.
        c.init_scene(scene="2a", layout=0, room=0, data_id = 444)
        c.add_overhead_camera({"x": 0, "y": 6.0, "z": -5.3}, target_object="a", images="cam")                
        commands = [{"$type": "set_floorplan_roof",
                                  "show": False}]
        resp = c.communicate(commands)
        images = get_data(resp=resp, d_type=Images)
        TDWUtils.save_images(images=images,
                             filename="20",
                             output_directory='./',
                             append_pass=False)
    finally:
        kill_tdw(docker_id)