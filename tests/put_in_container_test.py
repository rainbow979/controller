import random
import numpy as np
from typing import Tuple, Optional, List
from tdw.tdw_utils import TDWUtils
from sticky_mitten_avatar import StickyMittenAvatarController, Arm
from sticky_mitten_avatar.task_status import TaskStatus
from sticky_mitten_avatar.util import CONTAINER_SCALE, TARGET_OBJECT_MASS, CONTAINER_MASS


class PutInContainerTest(StickyMittenAvatarController):
    """
    This is a demo controller of simple navigation and arm articulation techniques.
    It is NOT an optimized solution! You will need to improve upon it.

    In this scene, there are target objects and containers.
    The avatar will pick up a random container, and then put objects into the container until it's full.
    """

    # Try to turn this many degrees per attempt at grasping an object.
    _D_THETA_GRASP = 15

    def __init__(self, port: int = 1071):
        super().__init__(port=port, launch_build=False, id_pass=False)
        self.container_ids: List[int] = []
        self.object_ids: List[int] = []

    def _get_scene_init_commands(self, scene: str = None, layout: int = None, room: int = -1) -> List[dict]:
        commands = super()._get_scene_init_commands()

        # This function includes low-level TDW commands that you won't need to use in an actual simulation.

        # Add containers.
        theta = 0
        num_containers = 8
        d_theta = 360 / num_containers
        pos = np.array([3.5, 0, 0])
        origin = np.array([0, 0, 0])
        for i in range(num_containers):
            container_position = TDWUtils.rotate_position_around(origin=origin, position=pos, angle=theta)
            container_id, container_commands = self._add_object("basket_18inx18inx12iin",
                                                                scale=CONTAINER_SCALE,
                                                                position=TDWUtils.array_to_vector3(container_position),
                                                                rotation={"x": 0, "y": random.uniform(-179, 179)})
            commands.extend(container_commands)
            commands.append({"$type": "set_mass",
                             "id": container_id,
                             "mass": CONTAINER_MASS})
            self.container_ids.append(container_id)
            theta += d_theta

        # Add target objects.
        num_objects = 10
        d_theta = 360 / num_objects
        theta = d_theta / 2
        pos = np.array([2, 0, 0])
        scale = {"x": 0.5, "y": 0.5, "z": 0.5}
        for i in range(num_objects):
            object_position = TDWUtils.rotate_position_around(origin=origin, position=pos, angle=theta)
            object_id, object_commands = self._add_object("jug05",
                                                          scale=scale,
                                                          position=TDWUtils.array_to_vector3(object_position),
                                                          rotation={"x": 0, "y": random.uniform(-179, 179)})
            commands.extend(object_commands)
            commands.append({"$type": "set_mass",
                             "id": object_id,
                             "mass": TARGET_OBJECT_MASS})
            self.object_ids.append(object_id)
            theta += d_theta

        return commands

    def _lift_arm(self, arm: Arm) -> None:
        """
        Lift the arm up.

        :param arm: The arm.
        """

        self.reach_for_target(arm=arm,
                              target={"x": -0.2 if arm == Arm.left else 0.2, "y": 0.4, "z": 0.3},
                              check_if_possible=False,
                              stop_on_mitten_collision=False)

    def _go_to_and_lift(self, object_ids: List[int], stopping_distance: float, object_type: str, arm: Arm = None) -> \
            Tuple[TaskStatus, int]:
        """
        Go to a random object. Try to grasp it and lift it up.

        :param object_ids: Choose an object from this list.
        :param object_type: A descriptor of the type of object used only for debugging.
        :param arm: If not None, the specified arm will try to grasp the object.
        :param stopping_distance:  Stop at this distance from the object.

        :return: Tuple: TaskStatus, and the object ID.
        """

        # Is the avatar already holding the object?
        for a in self.frame.held_objects:
            for object_id in self.frame.held_objects[a]:
                if object_id in object_ids:
                    print(f"Already holding a {object_type}.")
                    return TaskStatus.success, object_id
        object_id = random.choice(object_ids)

        # Lift up any arm that is holding an object.
        holding_arms = []
        for a in self.frame.held_objects:
            if len(self.frame.held_objects[a]) > 0:
                holding_arms.append(a)
        for a in holding_arms:
            self._lift_arm(arm=a)

        # Go to the object.
        self.go_to(object_id, move_stopping_threshold=stopping_distance)
        # Reset the arm positions after movement.
        for a in holding_arms:
            self.reset_arm(arm=a)
            self._lift_arm(arm=a)

        # Correct for a navigation error.
        d = np.linalg.norm(self.frame.avatar_transform.position - self.frame.object_transforms[object_id].position)
        for i in range(5):
            if d > 0.7:
                for a in holding_arms:
                    self.reset_arm(arm=a)
                    self._lift_arm(arm=a)
                self.go_to(object_id, move_stopping_threshold=stopping_distance)
            else:
                break

        for a in holding_arms:
            self._lift_arm(arm=a)

        # Pick up the object.
        success = self.grasp_and_lift(object_id=object_id, arm=arm)

        return TaskStatus.success if success else TaskStatus.failed_to_pick_up, object_id

    def lift_container(self) -> TaskStatus:
        """
        Go to a container. Grasp the container. Lift it up.

        :return: A TaskStatus; `success` when the avatar is holding a container.
        """

        status, object_id = self._go_to_and_lift(object_ids=self.container_ids, object_type="container",
                                                 stopping_distance=0.3)
        return status

    def lift_target_object(self) -> TaskStatus:
        """
        Go to a target object. Grasp the object. Lift it up.

        :return: A TaskStatus; `success` when the avatar is holding a target object.
        """

        # If the avatar is already holding a container, use the free mitten instead.
        # Otherwise, choose the mitten while trying to pick up the object.
        container_arm: Optional[Arm] = None
        for arm in self.frame.held_objects:
            for object_id in self.frame.held_objects[arm]:
                if object_id in self.container_ids:
                    container_arm = arm
        if container_arm is not None:
            arm = Arm.left if container_arm == Arm.right else Arm.right
        else:
            arm = None

        status, object_id = self._go_to_and_lift(object_ids=self.object_ids,
                                                 object_type="target object",
                                                 arm=arm,
                                                 stopping_distance=0.3)
        return status

    def grasp_and_lift(self, object_id: int, arm: Optional[Arm] = None) -> bool:
        """
        Repeatedly try to grasp a nearby object. If the object was grasped, lift it up.

        :param object_id: The ID of the target object.
        :param arm: Set the arm that should grasp and lift.

        :return: Tuple: True if the avatar grasped the object; the number of actions the avatar did.
        """

        def _turn_to_grasp(direction: int) -> bool:
            """
            Turn a bit, then try to grasp the object.
            This ends when the avatar has turned too far or if it grasps the object.

            :param direction: The direction to turn.

            :return: True if the avatar grasped the object.
            """

            theta = 0
            grasp_arm: Optional[Arm] = None
            # Try turning before giving up.
            # You can try adjusting this maximum.
            while theta < 90 and grasp_arm is None:
                # Try to grasp the object with each arm.
                for a in [Arm.left, Arm.right]:
                    if arm is not None and a != arm:
                        continue
                    s = self.grasp_object(object_id=object_id, arm=a)
                    if s == TaskStatus.success:
                        grasp_arm = a
                        break
                    else:
                        self.reset_arm(arm=a)
                if grasp_arm is None:
                    # Try turning some more.
                    s = self.turn_by(self._D_THETA_GRASP * direction)
                    # Failed to turn.
                    if s != TaskStatus.success:
                        return False
                    theta += self._D_THETA_GRASP
            if grasp_arm is not None:
                self._lift_arm(arm=grasp_arm)
            return grasp_arm is not None

        object_id = int(object_id)

        # Turn to face the object.
        self.turn_to(target=object_id)

        if arm is not None and arm == Arm.right:
            d = -1
        else:
            d = 1

        # Turn and grasp repeatedly.
        success = _turn_to_grasp(d)
        if success:
            print(f"Picked up {object_id}")
            return True

        # Reset the rotation.
        status = self.turn_by(-90)
        if status != TaskStatus.success:
            print(f"Failed to turn for some reason??")
            return False

        # Try turning the other way.
        d *= -1
        success = _turn_to_grasp(d)
        if success:
            print(f"Picked up {object_id}")
        else:
            print(f"Failed to pick up {object_id}")
        return success

    def try_put_in_container(self) -> TaskStatus:
        """
        Try to put an object in a container.

        :return: A TaskStatus indicating if the object is in the container and if not, why.
        """

        object_arm: Optional[Arm] = None
        container_arm: Optional[Arm] = None
        target_object_id: Optional[int] = None
        container_id: Optional[int] = None

        # Check if the avatar is holding a container and a target object.
        for arm in self.frame.held_objects:
            for object_id in self.frame.held_objects[arm]:
                if object_id in self.container_ids:
                    container_id = object_id
                    container_arm = arm
                elif object_id in self.object_ids:
                    target_object_id = object_id
                    object_arm = arm
        if container_arm is None:
            return TaskStatus.not_a_container
        if object_arm is None:
            return TaskStatus.failed_to_pick_up

        target_object_id = int(target_object_id)
        container_id = int(container_id)

        # Try to put the object in the container.
        status = self.put_in_container(object_id=target_object_id, container_id=container_id, arm=object_arm)
        print(f"Put in container: {status}")
        if status != TaskStatus.success:
            success = self.grasp_and_lift(object_id=container_id, arm=container_arm)
        else:
            success = True
            # Remove the object from the options of what can be picked up.
            self.object_ids.remove(target_object_id)

        return TaskStatus.success if success else TaskStatus.failed_to_pick_up

    def run(self) -> None:
        """
        For a number of trials, pick up an object, pick up a container, and put the object in the container.
        Try to reuse the container.
        """

        num_trials = 3
        num_successes = 0

        # Initialize the scene.
        self.init_scene()
        for i in range(num_trials):
            status = self.lift_container()
            if status != TaskStatus.success:
                continue
            status = self.lift_target_object()
            if status != TaskStatus.success:
                continue
            status = self.try_put_in_container()
            print(f"Tried to put object in container: {status}")
            if status == TaskStatus.success:
                num_successes += 1
        accuracy = float(num_successes) / num_trials
        print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    c = PutInContainerTest()
    c.run()
    c.end()
