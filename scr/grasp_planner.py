import numpy as np
from typing import Tuple

class GraspPlanner:
    def __init__(self, robot):
        self.robot = robot
        self.step_size = robot.step_size
        self.arm = robot.arm

    def calculate_targets(self, obj_position: np.ndarray, obj_width: float, grasp_clearance: float, class_name: str) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = obj_position
        max_y_offset = 0.2
        relay_left_y = min(y + 2 * obj_width + grasp_clearance, y + max_y_offset)
        relay_right_y = max(y - 2 * obj_width - grasp_clearance, y - max_y_offset)
        x_adjustment = 0.01 if class_name == 'can' else 0.0
        relay_left = np.array([x - grasp_clearance + x_adjustment, relay_left_y, z])
        relay_right = np.array([x - grasp_clearance + x_adjustment, relay_right_y, z])
        final_left = np.array([x - grasp_clearance + x_adjustment, y + obj_width * 0.3, z])
        final_right = np.array([x - grasp_clearance + x_adjustment, y - obj_width * 0.3, z])
        print(f"Using grasp_clearance={grasp_clearance}, x_adjustment={x_adjustment} for {class_name}")
        return relay_left, relay_right, final_left, final_right

    def move_to_relay(self, relay_left: np.ndarray, relay_right: np.ndarray) -> bool:
        left_current = self.arm.get_current_angles("left")
        right_current = self.arm.get_current_angles("right")
        left_target = self.robot.kinematics.inverse_kinematics(relay_left, left_current, "left")
        right_target = self.robot.kinematics.inverse_kinematics(relay_right, right_current, "right")
        left_trajectory = np.linspace(left_current, left_target, self.step_size)
        right_trajectory = np.linspace(right_current, right_target, self.step_size)
        left_reached = False
        right_reached = False
        for i in range(self.step_size):
            if not left_reached:
                self.arm.set_angles(left_trajectory[i], "left")
                if np.allclose(left_trajectory[i], left_target, atol=0.01):
                    left_reached = True
                    print("Left arm reached relay position")
            if not right_reached:
                self.arm.set_angles(right_trajectory[i], "right")
                if np.allclose(right_trajectory[i], right_target, atol=0.01):
                    right_reached = True
                    print("Right arm reached relay position")
            self.robot.step(self.robot.timestep)
            if left_reached and not right_reached:
                print("Left arm waiting for right arm...")
            elif right_reached and not left_reached:
                print("Right arm waiting for left arm...")
            elif left_reached and right_reached:
                print("Both arms reached relay positions")
                return True
        return left_reached and right_reached

    def execute_grasp(self, obj_position: np.ndarray, obj_width: float, grasp_clearance: float,
                      class_name: str, force_threshold: float = 2) -> bool:
        relay_left, relay_right, final_left, final_right = self.calculate_targets(obj_position, obj_width,
                                                                                  grasp_clearance, class_name)
        print(f"Relay targets - Left: {relay_left}, Right: {relay_right}")
        print(f"Final targets - Left: {final_left}, Right: {final_right}")

        print("Moving to relay positions")
        if not self.move_to_relay(relay_left, relay_right):
            print("Failed to reach relay positions")
            return False

        print("Moving both arms from relay to final grasp positions")
        left_current = self.arm.get_current_angles("left")
        right_current = self.arm.get_current_angles("right")
        left_target = self.robot.kinematics.inverse_kinematics(final_left, left_current, "left")
        right_target = self.robot.kinematics.inverse_kinematics(final_right, right_current, "right")
        left_trajectory = np.linspace(left_current, left_target, self.step_size)
        right_trajectory = np.linspace(right_current, right_target, self.step_size)

        left_contact = False
        right_contact = False
        for i in range(self.step_size):
            if not left_contact:
                self.arm.set_angles(left_trajectory[i], "left")
            if not right_contact:
                self.arm.set_angles(right_trajectory[i], "right")
            self.robot.step(self.robot.timestep)

            if not left_contact and self.arm.is_touch_detected("left", threshold=force_threshold):
                print(f"Left arm detected contact with force > {force_threshold} N at step {i}, stopping movement")
                self.arm.close_fingers_custom("left", force_threshold=force_threshold)
                left_contact = True

            if not right_contact and self.arm.is_touch_detected("right", threshold=force_threshold):
                print(f"Right arm detected contact with force > {force_threshold} N at step {i}, stopping movement")
                self.arm.close_fingers_custom("right", force_threshold=force_threshold)
                right_contact = True

            if left_contact and right_contact:
                print("Both arms have contacted the object, stopping movement")
                break

        if not (left_contact and right_contact):
            print(f"Reached final position without full contact: Left={left_contact}, Right={right_contact}")
            if self.arm.is_touch_detected("left", threshold=force_threshold) and not left_contact:
                print("Left arm detected contact at final position, stopping and closing fingers")
                self.arm.close_fingers_custom("left", force_threshold=force_threshold)
                left_contact = True
            if self.arm.is_touch_detected("right", threshold=force_threshold) and not right_contact:
                print("Right arm detected contact at final position, stopping and closing fingers")
                self.arm.close_fingers_custom("right", force_threshold=force_threshold)
                right_contact = True

        if left_contact and right_contact:
            print("Grasp completed, lifting object")
            self.lift_object(final_left, final_right)
            return True
        else:
            print(f"Grasp failed: Left contact={left_contact}, Right contact={right_contact}")
            self.arm.initialize_moves()
            return False

    def lift_object(self, final_left: np.ndarray, final_right: np.ndarray) -> None:
        lift_height = 0.06
        left_current = self.arm.get_current_angles("left")
        right_current = self.arm.get_current_angles("right")
        lift_left = np.array([final_left[0], final_left[1], final_left[2] + lift_height])
        lift_right = np.array([final_right[0], final_right[1], final_right[2] + lift_height])
        left_target = self.robot.kinematics.inverse_kinematics(lift_left, left_current, "left")
        right_target = self.robot.kinematics.inverse_kinematics(lift_right, right_current, "right")
        left_trajectory = np.linspace(left_current, left_target, self.step_size)
        right_trajectory = np.linspace(right_current, right_target, self.step_size)
        for i in range(self.step_size):
            self.arm.set_angles(left_trajectory[i], "left")
            self.arm.set_angles(right_trajectory[i], "right")
            self.robot.step(self.robot.timestep)
        print("Object lifted by 6 cm")
