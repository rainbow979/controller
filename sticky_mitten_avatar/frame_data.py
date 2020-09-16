from io import BytesIO
from PIL import Image
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from tdw.controller import Controller
from tdw.output_data import OutputData, Rigidbodies, Images, Transforms, CameraMatrices
from tdw.py_impact import PyImpact, AudioMaterial, Base64Sound, ObjectInfo
from tdw.tdw_utils import TDWUtils
from sticky_mitten_avatar.static_object_info import StaticObjectInfo
from sticky_mitten_avatar.avatars.avatar import Avatar
from sticky_mitten_avatar.util import get_data
from sticky_mitten_avatar.avatars import Arm
from sticky_mitten_avatar.transform import Transform


class FrameData:
    """
    Per-frame data that an avatar can use to decide what action to do next.

    Access this data from the [StickyMittenAvatarController](sma_controller.md):

    ```python
    from sticky_mitten_avatar import StickyMittenAvatarController, Arm

    c = StickyMittenAvatarController()
    c.init_scene()

    # Look towards the left arm.
    c.rotate_camera_by(pitch=70, yaw=-45)

    c.reach_for_target(target={"x": -0.2, "y": 0.21, "z": 0.385}, arm=Arm.left)

    # Save each image from the start of the most recent API action to the end.
    for frame in c.frames:
        frame.save_images(output_directory="dist")
    c.end()
    ```

    ***

    ## Fields

    - `object_transforms` The dictionary of [object transform data](transform.md). Key = the object ID.

    ```python
    # Print the position of each object per frame.
    for frame in c.frames:
        for object_id in frame.object_transforms:
            print(frame.object_transforms[object_id].position)
    ```

    - `audio` A list of tuples of audio generated by impacts. The first element in the tuple is a [`Base64Sound` object](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/py_impact.md#base64sound).
              The second element is the ID of the "target" (smaller) object.
    - `image_pass` Rendered image of the scene as a numpy array.
    - `id_pass` Image pass of object color segmentation as a numpy array.
    - `depth_pass` Image pass of depth values per pixel as a numpy array.
    - `avatar_transform` The [transform data](transform.md) of the avatar.
    - `avatar_body_part_transforms` The [transform data](transform.md) of each body part of the avatar. Key = body part ID.
    - `projection_matrix` The [camera projection matrix](https://github.com/threedworld-mit/tdw/blob/master/Documentation/api/output_data.md#cameramatrices) of the avatar's camera as a numpy array.
    - `camera_matrix` The [camera matrix](https://github.com/threedworld-mit/tdw/blob/master/Documentation/api/output_data.md#cameramatrices) of the avatar's camera as a numpy array.
    - `avatar_object_collisions` A dictionary of objects the avatar collided with. Key = body part ID. Value = A list of object IDs.

    ```python
    for frame in c.frames:
        for body_part_id in frame.avatar_object_collisions:
            body_part = c.static_avatar_info[body_part_id]
            object_ids = frame.avatar_object_collisions[body_part_id]
            for object_id in object_ids:
                print(body_part.name + " collided with object " + str(object_id))
    ```

    - `avatar_env_collisions`  A list of body part IDs that collided with the environment (such as a wall).

    ```python
    for frame in c.frames:
        for body_part_id in frame.avatar_env_collisions:
            body_part = c.static_avatar_info[body_part_id]
            print(body_part.name + " collided with the environment.")
    ```

    - `held_objects` A dictionary of IDs of objects held in each mitten. Key = arm:

    ```python
    from sticky_mitten_avatar import StickyMittenAvatarController, Arm

    c = StickyMittenAvatarController()

    # Your code here.

    # Prints all objects held by the left mitten.
    print(c.frames[-1].held_objects[Arm.left])
    ```

    ***

    ## Functions
    """

    _P = PyImpact(initial_amp=0.01)
    _SURFACE_MATERIAL: AudioMaterial = AudioMaterial.hardwood

    def __init__(self, resp: List[bytes], objects: Dict[int, StaticObjectInfo], avatar: Avatar):
        """
        :param resp: The response from the build.
        :param objects: Static object info per object. Key = the ID of the object in the scene.
        :param avatar: The avatar in the scene.
        """

        self._frame_count = Controller.get_frame(resp[-1])

        self.audio: List[Tuple[Base64Sound, int]] = list()
        collisions, env_collisions, rigidbodies = FrameData._P.get_collisions(resp=resp)

        # Record avatar collisions.
        if avatar is not None:
            self.avatar_object_collisions = avatar.collisions
            self.avatar_env_collisions = avatar.env_collisions
            self.held_objects = {Arm.left: avatar.frame.get_held_left(),
                                 Arm.right: avatar.frame.get_held_right()}
        else:
            self.avatar_object_collisions = None
            self.avatar_env_collisions = None
            self.held_objects = None

        # Get the object transform data.
        self.object_transforms: Dict[int, Transform] = dict()
        tr = get_data(resp=resp, d_type=Transforms)
        for i in range(tr.get_num()):
            o_id = tr.get_id(i)
            self.object_transforms[o_id] = Transform(position=np.array(tr.get_position(i)),
                                                     rotation=np.array(tr.get_rotation(i)),
                                                     forward=np.array(tr.get_forward(i)))

        # Get camera matrix data.
        matrices = get_data(resp=resp, d_type=CameraMatrices)
        self.projection_matrix = matrices.get_projection_matrix()
        self.camera_matrix = matrices.get_camera_matrix()

        # Get the transform data of the avatar.
        self.avatar_transform = Transform(position=np.array(avatar.frame.get_position()),
                                          rotation=np.array(avatar.frame.get_rotation()),
                                          forward=np.array(avatar.frame.get_forward()))
        self.avatar_body_part_transforms: Dict[int, Transform] = dict()
        for i in range(avatar.frame.get_num_body_parts()):
            self.avatar_body_part_transforms[avatar.frame.get_body_part_id(i)] = Transform(
                position=np.array(avatar.frame.get_body_part_position(i)),
                rotation=np.array(avatar.frame.get_body_part_rotation(i)),
                forward=np.array(avatar.frame.get_body_part_forward(i)))

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
                if collider_id in avatar.body_parts_static:
                    collider_info = avatar.body_parts_static[collider_id].audio
            if collidee_id in objects:
                collidee_info = objects[collidee_id].audio
            # Check if the object is a body part.
            else:
                if collidee_id in avatar.body_parts_static:
                    collidee_info = avatar.body_parts_static[collidee_id].audio

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
                audio = FrameData._P.get_sound(coll, rigidbodies, 1, FrameData._SURFACE_MATERIAL.name, collider_id,
                                               collider_info.material.name, 0.01)
                self.audio.append((audio, collider_id))
        # Get the image data.
        self.id_pass: Optional[np.array] = None
        self.depth_pass: Optional[np.array] = None
        self.image_pass: Optional[np.array] = None
        for i in range(0, len(resp) - 1):
            if OutputData.get_data_type_id(resp[i]) == "imag":
                images = Images(resp[i])
                for j in range(images.get_num_passes()):
                    if images.get_pass_mask(j) == "_id":
                        self.id_pass = images.get_image(j)
                    elif images.get_pass_mask(j) == "_depth_simple":
                        self.depth_pass = images.get_image(j)
                    elif images.get_pass_mask(j) == "_img":
                        self.image_pass = images.get_image(j)

    @staticmethod
    def set_surface_material(surface_material: AudioMaterial) -> None:
        """
        Set the surface material of the scene.

        :param surface_material: The floor's [audio material](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/py_impact.md#audiomaterialenum).
        """

        FrameData._SURFACE_MATERIAL = surface_material

    def save_images(self, output_directory: Union[str, Path]) -> None:
        """
        Save the ID pass (segmentation colors) and the depth pass to disk.
        Images will be named: `[frame_number]_[pass_name].[extension]`
        For example, the depth pass on the first frame will be named: `00000000_depth.png`
        The image pass is a jpg file and the other passes are png files.

        :param output_directory: The directory that the images will be saved to.
        """

        if isinstance(output_directory, str):
            output_directory = Path(output_directory)
        if not output_directory.exists():
            output_directory.mkdir(parents=True)
        prefix = TDWUtils.zero_padding(self._frame_count, 8)
        # Save each image.
        for image, pass_name, ext in zip([self.image_pass, self.id_pass, self.depth_pass], ["img", "id", "depth"],
                                       ["jpg", "png", "png"]):
            p = output_directory.joinpath(f"{prefix}_{pass_name}.{ext}")
            with p.open("wb") as f:
                f.write(image)

    def get_pil_images(self) -> dict:
        """
        Convert each image pass to PIL images.

        :return: A dictionary of PIL images. Key = the name of the pass (img, id, depth)
        """

        print(type(Image.open(BytesIO(self.image_pass))))

        return {"img": Image.open(BytesIO(self.image_pass)),
                "id": Image.open(BytesIO(self.id_pass)),
                "depth": Image.open(BytesIO(self.depth_pass))}

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
