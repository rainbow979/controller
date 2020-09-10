from io import BytesIO
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from tdw.output_data import OutputData, Rigidbodies, Images, Transforms
from tdw.py_impact import PyImpact, AudioMaterial, Base64Sound, ObjectInfo
from tdw.tdw_utils import TDWUtils
from sticky_mitten_avatar.static_object_info import StaticObjectInfo
from sticky_mitten_avatar.avatars.avatar import Avatar
from sticky_mitten_avatar.util import get_data


class AvatarCollisions:
    """
    Collisions between an avatar and other objects and the environment on this frame.
    To get the segmentation color and the name of each body part, see `StickyMittenAvatarController.static_avatar_info`.

    ```python
    from sticky_mitten_avatar import StickyMittenAvatarController

    c = StickyMittenAvatarController()
    c.init_scene()

    # Your code here.

    for avatar_id in c.frame.avatar_collisions:
        # Get each body part that collided with an object.
        for body_part_id in c.frame.avatar_collisions[avatar_id].objects:
            body_part = c.static_avatar_info[avatar_id][body_part_id]
            print(body_part.color)
            print(body_part.name)
            for object_id in c.frame.avatar_collisions[avatar_id].objects[body_part_id]:
                print(object_id)
        # Get each body part that collided with the environment.
        for body_part_id in c.frame.avatar_collisions[avatar_id].env:
            body_part = c.static_avatar_info[avatar_id][body_part_id]
            print(body_part.name + " collided with the environment.")
    ```

    Fields:

    - `objects` A dictionary of objects the avatar collided with. Key = body part ID. Value = A list of object IDs.
    - `env` A list of body part IDs that collided with the environment (such as a wall).
    """

    def __init__(self, avatar: Avatar):
        """
        :param avatar: The avatar.
        """

        self.objects = avatar.collisions
        self.env = avatar.env_collisions


class FrameData:
    """
    Per-frame data that an avatar can use to decide what action to do next.

    Access this data from the StickyMittenAvatarController:

    ```python
    from sticky_mitten_avatar import StickyMittenAvatarController

    c = StickyMittenAvatarController()
    c.init_scene()

    print(c.frame.positions)
    ```

    Fields:

    - `positions` The dictionary of object positions. Key = the object ID. Value = the position as a numpy array.
    - `audio` A list of tuples of audio generated by impacts. The first element in the tuple is a [`Base64Sound` object](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/py_impact.md#base64sound).
              The second element is the ID of the "target" (smaller) object.
    - `id_pass` Image pass of object color segmentation as a numpy array.
    - `depth_pass` Image pass of depth values per pixel as a numpy array. Can be None.
    - `avatar_collisions` Collisions per avatar between any of its body parts and another object or the environment.
                          See `AvatarCollisions` for more information.
    """

    _P = PyImpact(initial_amp=0.03)
    _FRAME_COUNT = 0

    def __init__(self, resp: List[bytes], objects: Dict[int, StaticObjectInfo], surface_material: AudioMaterial,
                 avatars: Dict[str, Avatar]):
        """
        :param resp: The response from the build.
        :param objects: Static object info per object. Key = the ID of the object in the scene.
        :param surface_material: The floor's [audio material](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/py_impact.md#audiomaterialenum).
        :param avatars: Each avatar in the scene.
        """

        FrameData._FRAME_COUNT += 1

        self.audio: List[Tuple[Base64Sound, int]] = list()
        collisions, env_collisions, rigidbodies = FrameData._P.get_collisions(resp=resp)

        # Record avatar collisions.
        self.avatar_collisions: Dict[str, AvatarCollisions] = dict()
        for avatar_id in avatars:
            self.avatar_collisions[avatar_id] = AvatarCollisions(avatar=avatars[avatar_id])

        self.positions: Dict[int, np.array] = dict()
        tr = get_data(resp=resp, d_type=Transforms)
        for i in range(tr.get_num()):
            o_id = tr.get_id(i)
            self.positions[o_id] = np.array(tr.get_position(i))

        # Get the audio of each collision.
        for coll in collisions:
            if not FrameData._P.is_valid_collision(coll):
                continue

            collider_id = coll.get_collider_id()
            collidee_id = coll.get_collidee_id()

            collider_info: Optional[ObjectInfo] = None
            collidee_info: Optional[ObjectInfo] = None

            if collider_id in objects:
                collider_info = objects[collider_id].audio
            # Check if the object is a body part.
            else:
                for avatar_id in avatars:
                    if collider_id in avatars[avatar_id].body_parts_static:
                        collider_info = avatars[avatar_id].body_parts_static[collider_id].audio
            if collidee_id in objects:
                collidee_info = objects[collidee_id].audio
            # Check if the object is a body part.
            else:
                for avatar_id in avatars:
                    if collidee_id in avatars[avatar_id].body_parts_static:
                        collidee_info = avatars[avatar_id].body_parts_static[collidee_id].audio

            # If either object isn't a cached object, don't try to add audio.
            if collider_info is None or collidee_info is None:
                continue

            if collider_info.mass < collidee_info.mass:
                target_id = collider_id
                target_amp = collider_info.amp
                target_mat = collider_info.material.name
                other_id = collidee_id
                other_amp = collidee_info.amp
                other_mat = collider_info.material.name
            else:
                target_id = collidee_id
                target_amp = collidee_info.amp
                target_mat = collidee_info.material.name
                other_id = collider_id
                other_amp = collider_info.amp
                other_mat = collider_info.material.name
            rel_amp = other_amp / target_amp
            audio = FrameData._P.get_sound(coll, rigidbodies, other_id, other_mat, target_id, target_mat, rel_amp)
            self.audio.append((audio, target_id))
        # Get the audio of each environment collision.
        for coll in env_collisions:
            collider_id = coll.get_object_id()
            v = FrameData._get_velocity(rigidbodies, collider_id)
            if (v is not None) and (v > 0):
                collider_info = objects[collider_id].audio
                audio = FrameData._P.get_sound(coll, rigidbodies, 1, surface_material.name, collider_id,
                                               collider_info.material.name, 0.01)
                self.audio.append((audio, collider_id))
        # Get the image data.
        self.id_pass: Optional[np.array] = None
        self.depth_pass: Optional[np.array] = None

        for i in range(0, len(resp) - 1):
            if OutputData.get_data_type_id(resp[i]) == "imag":
                images = Images(resp[i])
                for j in range(images.get_num_passes()):
                    if images.get_pass_mask(j) == "_id":
                        self.id_pass = images.get_image(j)
                    elif images.get_pass_mask(j) == "_depth_simple":
                        self.depth_pass = images.get_image(j)

    def save_images(self, output_directory: Union[str, Path]) -> None:
        """
        Save the ID pass (segmentation colors) and the depth pass to disk.
        Images will be named: `[frame_number]_[pass_name].png`
        For example, the depth pass on the first frame will be named: `00000000_depth.png`

        :param output_directory: The directory that the images will be saved to.
        """

        if isinstance(output_directory, str):
            output_directory = Path(output_directory)
        if not output_directory.exists():
            output_directory.mkdir(parents=True)
        prefix = TDWUtils.zero_padding(FrameData._FRAME_COUNT, 8)
        # Save each image.
        for image, pass_name in zip([self.id_pass, self.depth_pass], ["id", "depth"]):
            p = output_directory.joinpath(f"{prefix}_{pass_name}.png")
            with p.open("wb") as f:
                f.write(image)

    def get_pil_images(self) -> tuple:
        """
        Convert the ID pass (segmentation colors) and the depth pass to PIL images.

        :return: Tuple: (ID pass, depth pass) as PIL images.
        """

        return Image.open(BytesIO(self.id_pass)), Image.open(BytesIO(self.depth_pass))

    @staticmethod
    def _get_velocity(rigidbodies: Rigidbodies, o_id: int) -> float:
        """
        :param rigidbodies: The rigidbody data.
        :param o_id: The ID of the object.

        :return: The velocity magnitude of the object.
        """

        for i in range(rigidbodies.get_num()):
            if rigidbodies.get_id(i) == o_id:
                return np.linalg.norm(rigidbodies.get_velocity(i))
