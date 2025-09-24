from controller import Robot, Camera, PositionSensor, Motion
from ultralytics import YOLO
import cv2
import numpy as np
import os
from typing import Tuple, Optional, Dict, List

class Nao(Robot):
    def __init__(self, model_path: str):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        # Camera initialization
        self.camera = self.getDevice('CameraBottom')
        self.camera.enable(self.timestep)
        self.image_width = self.camera.getWidth()
        self.image_height = self.camera.getHeight()

        # Head sensors and motors
        self.head_yaw_sensor = self.getDevice('HeadYawS')
        self.head_pitch_sensor = self.getDevice('HeadPitchS')
        self.head_yaw_sensor.enable(self.timestep)
        self.head_pitch_sensor.enable(self.timestep)
        self.head_yaw_motor = self.getDevice('HeadYaw')
        self.head_pitch_motor = self.getDevice('HeadPitch')

        # Arm and hand motors/sensors
        self.left_arm_motors = {f"L{name}": self.getDevice(f"L{name}") for name in
                                ["ShoulderPitch", "ShoulderRoll", "ElbowYaw", "ElbowRoll", "WristYaw"]}
        self.right_arm_motors = {f"R{name}": self.getDevice(f"R{name}") for name in
                                 ["ShoulderPitch", "ShoulderRoll", "ElbowYaw", "ElbowRoll", "WristYaw"]}
        self.left_arm_sensors = {f"L{name}S": self.getDevice(f"L{name}S") for name in
                                 ["ShoulderPitch", "ShoulderRoll", "ElbowYaw", "ElbowRoll", "WristYaw"]}
        self.right_arm_sensors = {f"R{name}S": self.getDevice(f"R{name}S") for name in
                                  ["ShoulderPitch", "ShoulderRoll", "ElbowYaw", "ElbowRoll", "WristYaw"]}
        for sensor in list(self.left_arm_sensors.values()) + list(self.right_arm_sensors.values()):
            if isinstance(sensor, PositionSensor):
                sensor.enable(self.timestep)
        self.left_phalanxes = {f"LPhalanx{i}": self.getDevice(f"LPhalanx{i}") for i in range(1, 9)}
        self.right_phalanxes = {f"RPhalanx{i}": self.getDevice(f"RPhalanx{i}") for i in range(1, 9)}

        # Touch sensors initialization
        self.left_touch_sensor = self.getDevice('Ltouchsensor')
        self.right_touch_sensor = self.getDevice('Rtouchsensor')
        self.left_touch_sensor.enable(self.timestep)
        self.right_touch_sensor.enable(self.timestep)

        # Motion files initialization with validation
        self.motions = {}
        base_path = "C:/Program Files/Webots/projects/robots/softbank/nao/motions/"
        motion_files = ["SideStepLeft.motion", "SideStepRight.motion", "SideStepLeft08.motion",
                        "SideStepRight08.motion", "Forwards.motion", "ForwardsSmall.motion", "Backwards.motion"]
        for motion_file in motion_files:
            motion_name = motion_file.split('.')[0]
            motion = Motion(os.path.join(base_path, motion_file))
            if motion.isValid():
                self.motions[motion_name] = motion
                print(f"Loaded {motion_name} successfully, duration: {motion.getDuration()} ms")
            else:
                print(f"Failed to load {motion_name}, skipping")
                del motion

        # Vision system initialization
        self.hfov = 60.97
        self.vfov = 47.64
        self.horizontal_focal_length = (self.image_width / 2) / np.tan(np.radians(self.hfov / 2))
        self.vertical_focal_length = (self.image_height / 2) / np.tan(np.radians(self.vfov / 2))
        self.object_reference = {
            'can': {'height': 0.1222, 'diameter': 0.03175, 'grasp_clearance': 0.06},
            'ball': {'diameter': 0.0325, 'grasp_clearance': 0.03}
        }
        self.camera_x = 0.05071
        self.camera_z = 0.01774
        self.torso_z_offset = 0.1265
        try:
            self.yolo_model = YOLO(model_path)
        except Exception as e:
            self.yolo_model = None

        # Head controller initialization
        self.yaw_limits = (-2.09, 2.09)
        self.pitch_limits = (-0.67, 0.51)

        # Kinematics controller initialization
        self.L = [0.098, 0.100, 0.015, 0.105, 0.05595, 0.05775, 0.01231]
        self.joint_limits = {
            "LShoulderPitch": (-2.0857, 2.0857), "LShoulderRoll": (-0.314159, 1.3265),
            "LElbowYaw": (-2.0857, 2.0857), "LElbowRoll": (-1.5446, -0.0349),
            "LWristYaw": (-1.8239, 1.8239), "RShoulderPitch": (-2.0857, 2.0857),
            "RShoulderRoll": (-1.3265, 0.314159), "RElbowYaw": (-2.0857, 2.0857),
            "RElbowRoll": (0.0349, 1.5446), "RWristYaw": (-1.8239, 1.8239)
        }
        self.phalanx_limits = (0.0, 1.0)

        # Grasp planner initialization
        self.step_size = 50

    def continuous_move(self, motion_name: str, steps: int, reverse: bool = False) -> bool:
        """執行指定次數的連續移動"""
        motion = self.motions.get(motion_name)
        if not motion or not motion.isValid():
            print(f"Motion {motion_name} not available or invalid")
            return False

        motion.setReverse(reverse)
        motion.setLoop(True)
        motion.play()
        print(f"Starting continuous {motion_name}, steps={steps}, reverse={reverse}")

        total_steps = steps * (motion.getDuration() // self.timestep)  # 估計總模擬步數
        for i in range(total_steps):
            self.step(self.timestep)
            if i % 10 == 0 and (self.is_touch_detected("left", 1.0) or self.is_touch_detected("right", 1.0)):
                motion.stop()
                print("Touch detected, stopping motion")
                self.initialize_arms()
                return False
        motion.stop()
        print(f"Completed {motion_name} for {steps} steps")
        return True

    def play_motion(self, motion_name: str, reverse: bool = False) -> bool:
        """單次播放 motion，支援反向"""
        motion = self.motions.get(motion_name)
        if not motion or not motion.isValid():
            print(f"Motion {motion_name} not available or invalid")
            return False
        motion.setReverse(reverse)
        motion.play()
        while not motion.isOver():
            self.step(self.timestep)
        print(f"Completed {motion_name}, reverse={reverse}")
        return True

    def update_position(self, frame):
        """更新位置並重新對準"""
        if frame is None:
            print("Failed to get camera frame in update_position")
            return None
        head_angles = self.get_head_angles()
        processed_frame, detected_objects = self.process_frame(frame, head_angles)
        if not detected_objects:
            print("No objects detected during update_position")
            return None
        closest = min(detected_objects, key=lambda x: x['distance'])
        self.align_to_target(closest['center'][0], closest['center'][1])
        return closest
    # General methods
    def is_touch_detected(self, side: str, threshold: float = 0.3) -> bool:
        sensor = self.left_touch_sensor if side == "left" else self.right_touch_sensor
        force_value = sensor.getValue()
        print(f"{side} touch sensor force value: {force_value:.3f} N")
        return force_value > threshold

    def get_camera_image(self) -> Optional[np.ndarray]:
        image = self.camera.getImage()
        if image:
            return cv2.cvtColor(np.frombuffer(image, np.uint8).reshape((self.image_height, self.image_width, 4)),
                                cv2.COLOR_RGBA2BGR)
        return None

    def get_head_angles(self) -> Tuple[float, float]:
        return self.head_yaw_sensor.getValue(), self.head_pitch_sensor.getValue()

    # Vision methods
    def process_frame(self, frame: np.ndarray, head_angles: Tuple[float, float]) -> Tuple[Optional[np.ndarray], List[Dict]]:
        if frame is None or self.yolo_model is None:
            return None, []
        frame_resized = cv2.resize(frame, (self.image_width, self.image_height))
        results = self.yolo_model(frame_resized, conf=0.6)[0]
        detected_objects = []
        for detection in results.boxes.data:
            x1, y1, x2, y2, conf, class_id = map(float, detection[:6])
            class_name = results.names[int(class_id)].lower()
            if class_name not in self.object_reference:
                continue
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            position, distance = self._calculate_coordinates(center_x, center_y, y2 - y1, class_name, head_angles)
            detected_objects.append({
                'class_name': class_name,
                'center': (center_x, center_y),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': conf,
                'position': position,
                'distance': distance,
                'grasp_clearance': self.object_reference[class_name]['grasp_clearance'],
                'width': self.object_reference[class_name]['diameter']
            })
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"{class_name}: {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame_resized, f"Pos: {position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}",
                        (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame_resized, detected_objects

    def _calculate_coordinates(self, center_x: float, center_y: float, object_height_pixels: float,
                              class_name: str, head_angles: Tuple[float, float]) -> Tuple[np.ndarray, float]:
        real_height = self.object_reference[class_name].get('height', self.object_reference[class_name]['diameter'])
        x_offset = center_x - self.image_width / 2
        y_offset = center_y - self.image_height / 2
        horizontal_angle = np.arctan2(x_offset, self.horizontal_focal_length)
        vertical_angle = np.arctan2(y_offset, self.vertical_focal_length)
        distance = (self.vertical_focal_length * real_height) / object_height_pixels
        x_cam = distance * np.tan(horizontal_angle)
        y_cam = distance * np.tan(vertical_angle)
        z_cam = distance
        position = self._transform_to_torso_coordinates(x_cam, y_cam, z_cam, head_angles)
        if 'height' in self.object_reference[class_name]:
            position[2] -= self.object_reference[class_name]['height'] * 0.3
        else:
            position[2] -= self.object_reference[class_name]['diameter'] / 2
        print(f"Z-coordinate corrected for {class_name}: {position[2]}")
        return position, distance

    def _transform_to_torso_coordinates(self, x_cam: float, y_cam: float, z_cam: float,
                                       head_angles: Tuple[float, float]) -> np.ndarray:
        yaw, pitch = head_angles
        ntcdistance = np.sqrt(self.camera_x ** 2 + self.camera_z ** 2)
        t01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.torso_z_offset], [0, 0, 0, 1]])
        t12 = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        t23 = np.array([[np.cos(pitch), 0, np.sin(pitch), 0], [0, 1, 0, 0], [-np.sin(pitch), 0, np.cos(pitch), 0], [0, 0, 0, 1]])
        t3camera = np.array([[1, 0, 0, self.camera_x + ntcdistance * np.cos(pitch)],
                             [0, 1, 0, 0],
                             [0, 0, 1, self.camera_z - ntcdistance * np.sin(pitch) - 0.11],
                             [0, 0, 0, 1]])
        t0camera = t01 @ t12 @ t23 @ t3camera
        camera_coords = np.array([z_cam, -x_cam, -y_cam, 1])
        torso_coords = t0camera @ camera_coords
        return torso_coords[:3]

    # Head control methods
    def set_head_angles(self, yaw_deg: float, pitch_deg: float) -> None:
        yaw_rad = np.radians(np.clip(yaw_deg, -119.5, 119.5))
        pitch_rad = np.radians(np.clip(pitch_deg, -29.5, 38.5))
        if self.yaw_limits[0] <= yaw_rad <= self.yaw_limits[1] and self.pitch_limits[0] <= pitch_rad <= self.pitch_limits[1]:
            self.head_yaw_motor.setPosition(yaw_rad)
            self.head_pitch_motor.setPosition(pitch_rad)

    def align_to_target(self, center_x: float, center_y: float) -> bool:
        current_yaw, current_pitch = self.get_head_angles()
        x_offset = center_x - self.image_width / 2
        y_offset = center_y - self.image_height / 2
        horizontal_angle = np.arctan2(x_offset, self.horizontal_focal_length)
        vertical_angle = np.arctan2(y_offset, self.vertical_focal_length)
        required_yaw = current_yaw - horizontal_angle
        required_pitch = current_pitch + vertical_angle
        self.set_head_angles(np.degrees(required_yaw), np.degrees(required_pitch))
        yaw_error = abs(horizontal_angle)
        pitch_error = abs(vertical_angle)
        print(f"Object center: ({center_x}, {center_y}), Yaw error: {yaw_error:.4f} rad, Pitch error: {pitch_error:.4f} rad")
        return yaw_error < 0.1 and pitch_error < 0.1
    # Kinematics methods
    def forward_kinematics(self, angles: np.ndarray, arm_side: str) -> np.ndarray:
        if not isinstance(angles, np.ndarray) or angles.shape != (5,):
            raise ValueError("Angles must be a numpy array of shape (5,)")
        l1, l2, l3, l4, l5, l6, l7 = self.L
        if arm_side == "left":
            t_shoulder = np.array([[1, 0, 0, 0], [0, 1, 0, l1], [0, 0, 1, l2], [0, 0, 0, 1]])
            t1 = np.array([[np.cos(angles[0]), -np.sin(angles[0]), 0, 0], [0, 0, 1, 0],
                           [-np.sin(angles[0]), -np.cos(angles[0]), 0, 0], [0, 0, 0, 1]])
            t2 = np.array([[-np.sin(angles[1]), -np.cos(angles[1]), 0, 0], [0, 0, -1, 0],
                           [np.cos(angles[1]), -np.sin(angles[1]), 0, 0], [0, 0, 0, 1]])
            t3 = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0, l3], [0, 0, -1, -l4],
                           [np.sin(angles[2]), np.cos(angles[2]), 0, 0], [0, 0, 0, 1]])
            t4 = np.array([[np.cos(angles[3]), -np.sin(angles[3]), 0, 0], [0, 0, 1, 0],
                           [-np.sin(angles[3]), -np.cos(angles[3]), 0, 0], [0, 0, 0, 1]])
            t5 = np.array([[np.cos(angles[4]), -np.sin(angles[4]), 0, 0], [0, 0, -1, -l5],
                           [np.sin(angles[4]), np.cos(angles[4]), 0, 0], [0, 0, 0, 1]])
            t_hand = np.array([[0, 1, 0, 0], [0, 0, 1, -l7], [1, 0, 0, l6], [0, 0, 0, 1]])
            t = t_shoulder @ t1 @ t2 @ t3 @ t4 @ t5 @ t_hand
        else:
            t_shoulder = np.array([[1, 0, 0, 0], [0, 1, 0, -l1], [0, 0, 1, l2], [0, 0, 0, 1]])
            t1 = np.array([[np.cos(angles[0]), -np.sin(angles[0]), 0, 0], [0, 0, 1, 0],
                           [-np.sin(angles[0]), -np.cos(angles[0]), 0, 0], [0, 0, 0, 1]])
            t2 = np.array([[np.sin(angles[1]), np.cos(angles[1]), 0, 0], [0, 0, -1, 0],
                           [-np.cos(angles[1]), np.sin(angles[1]), 0, 0], [0, 0, 0, 1]])
            t3 = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0, l3], [0, 0, 1, l4],
                           [-np.sin(angles[2]), -np.cos(angles[2]), 0, 0], [0, 0, 0, 1]])
            t4 = np.array([[np.cos(angles[3]), -np.sin(angles[3]), 0, 0], [0, 0, -1, 0],
                           [np.sin(angles[3]), np.cos(angles[3]), 0, 0], [0, 0, 0, 1]])
            t5 = np.array([[np.cos(angles[4]), -np.sin(angles[4]), 0, 0], [0, 0, 1, l5],
                           [-np.sin(angles[4]), -np.cos(angles[4]), 0, 0], [0, 0, 0, 1]])
            t_hand = np.array([[0, -1, 0, 0], [0, 0, -1, l7], [1, 0, 0, l6], [0, 0, 0, 1]])
            t = t_shoulder @ t1 @ t2 @ t3 @ t4 @ t5 @ t_hand
        return t[:3, 3]

    def inverse_kinematics(self, target_pos: np.ndarray, init_angles: np.ndarray, arm_side: str,
                          learning_rate: float = 0.01, max_iterations: int = 1000,
                          tolerance: float = 1e-5) -> np.ndarray:
        if not isinstance(init_angles, np.ndarray) or init_angles.shape != (5,):
            raise ValueError("Initial angles must be a numpy array of shape (5,)")
        angles = init_angles.copy()
        lambda_reg = 1e-3
        for i in range(max_iterations):
            current_position = self.forward_kinematics(angles, arm_side)
            error = target_pos - current_position
            if np.linalg.norm(error) < tolerance:
                print(f"{arm_side} arm IK converged at iteration {i}, error: {np.linalg.norm(error)}")
                break
            jacobian = np.zeros((3, 5))
            delta_angle = 1e-5
            for j in range(5):
                perturbed_angles = angles.copy()
                perturbed_angles[j] += delta_angle
                perturbed_position = self.forward_kinematics(perturbed_angles, arm_side)
                jacobian[:, j] = (perturbed_position - current_position) / delta_angle
            jacobian_reg = jacobian.T @ jacobian + lambda_reg * np.eye(5)
            delta_angles = np.linalg.inv(jacobian_reg) @ jacobian.T @ error
            angles += learning_rate * delta_angles
        return angles

    def clamp_angle(self, angle: float, joint_name: str) -> float:
        min_angle, max_angle = self.joint_limits.get(joint_name, self.phalanx_limits)
        return max(min(angle, max_angle), min_angle)

    # Arm control methods
    def set_angles(self, angles: np.ndarray, arm_side: str) -> None:
        motors = self.left_arm_motors if arm_side == "left" else self.right_arm_motors
        joint_names = list(motors.keys())
        clamped_angles = [self.clamp_angle(angle, name) for angle, name in zip(angles, joint_names)]
        for angle, name in zip(clamped_angles, joint_names):
            motors[name].setPosition(angle)

    def get_current_angles(self, arm_side: str) -> np.ndarray:
        sensors = self.left_arm_sensors if arm_side == "left" else self.right_arm_sensors
        return np.array([sensor.getValue() for sensor in sensors.values()])

    def move_to(self, target_pos: np.ndarray, arm_side: str, steps: int = 50) -> None:
        current_angles = self.get_current_angles(arm_side)
        target_angles = self.inverse_kinematics(target_pos, current_angles, arm_side)
        trajectory = np.linspace(current_angles, target_angles, steps)
        for angles in trajectory:
            self.set_angles(angles, arm_side)
            self.step(self.timestep)

    def open_fingers(self, arm_side: str) -> None:
        phalanxes = self.left_phalanxes if arm_side == "left" else self.right_phalanxes
        for phalanx in phalanxes.values():
            phalanx.setPosition(self.phalanx_limits[1])
        print(f"{arm_side} hand fingers opened")

    def close_fingers_custom(self, arm_side: str, target_position: float = 0.85, steps: int = 60,
                             force_threshold: float = 2) -> None:
        phalanxes = self.left_phalanxes if arm_side == "left" else self.right_phalanxes
        start_pos = self.phalanx_limits[1]  # 手指初始位置（完全打開）
        step_size = (start_pos - target_position) / steps
        finger_groups = [[1, 2, 3], [4, 5, 6], [7, 8]]  # 各關節之手指分組

        print(f"{arm_side} hand: Starting finger closing to position {target_position}")

        # 逐步關閉手指，直到檢測到足夠的力
        for step in range(steps + 1):
            if self.is_touch_detected(arm_side, threshold=force_threshold):
                print(
                    f"{arm_side} hand: Force detected above {force_threshold} N, stopping arm movement and closing fingers")
                break

            current_pos = start_pos - step * step_size
            for group in finger_groups:
                for i in group:
                    phalanx_key = f"{'L' if arm_side == 'left' else 'R'}Phalanx{i}"
                    phalanxes[phalanx_key].setPosition(current_pos)
            self.step(self.timestep)

        # 碰到後合起手指到目標位置
        print(f"{arm_side} hand: Closing fingers to final position {target_position}")
        for group in finger_groups:
            for i in group:
                phalanx_key = f"{'L' if arm_side == 'left' else 'R'}Phalanx{i}"
                phalanxes[phalanx_key].setPosition(target_position)

        for _ in range(10):  # 額外step以穩定手指合起
            self.step(self.timestep)

        print(f"{arm_side} hand: Fingers closed to position {target_position}")

    def initialize_arms(self) -> None:
        print("Initializing both arms simultaneously")
        left_initial_angles = np.array([1.2, 0.5, -0.9, -1.54462, -0.5])
        right_initial_angles = np.array([1.2, -0.5, 0.9, 1.54462, 0.5])
        self.set_angles(left_initial_angles, "left")
        self.set_angles(right_initial_angles, "right")
        self.open_fingers("left")
        self.open_fingers("right")
        for _ in range(100):
            self.step(self.timestep)
        print("Both arms initialized")

    # Grasp planner methods
    def calculate_targets(self, obj_position: np.ndarray, obj_width: float, grasp_clearance: float, class_name: str) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = obj_position
        max_y_offset = 0.2
        relay_left_y = min(y + 2 * obj_width + grasp_clearance, y + max_y_offset)
        relay_right_y = max(y - 2 * obj_width - grasp_clearance, y - max_y_offset)
        x_adjustment = -0.03 if class_name == 'can' else 0.0
        relay_left = np.array([x - grasp_clearance + x_adjustment, relay_left_y, z])
        relay_right = np.array([x - grasp_clearance + x_adjustment, relay_right_y, z])
        final_left = np.array([x - grasp_clearance + x_adjustment, y + obj_width * 0.3, z])
        final_right = np.array([x - grasp_clearance + x_adjustment, y - obj_width * 0.3, z])
        print(f"Using grasp_clearance={grasp_clearance}, x_adjustment={x_adjustment} for {class_name}")
        return relay_left, relay_right, final_left, final_right

    def move_to_relay(self, relay_left: np.ndarray, relay_right: np.ndarray) -> bool:
        left_current = self.get_current_angles("left")
        right_current = self.get_current_angles("right")
        left_target = self.inverse_kinematics(relay_left, left_current, "left")
        right_target = self.inverse_kinematics(relay_right, right_current, "right")
        left_trajectory = np.linspace(left_current, left_target, self.step_size)
        right_trajectory = np.linspace(right_current, right_target, self.step_size)
        left_reached = False
        right_reached = False
        for i in range(self.step_size):
            if not left_reached:
                self.set_angles(left_trajectory[i], "left")
                if np.allclose(left_trajectory[i], left_target, atol=0.01):
                    left_reached = True
                    print("Left arm reached relay position")
            if not right_reached:
                self.set_angles(right_trajectory[i], "right")
                if np.allclose(right_trajectory[i], right_target, atol=0.01):
                    right_reached = True
                    print("Right arm reached relay position")
            self.step(self.timestep)
            if left_reached and not right_reached:
                print("Left arm waiting for right arm...")
            elif right_reached and not left_reached:
                print("Right arm waiting for left arm...")
            elif left_reached and right_reached:
                print("Both arms reached relay positions")
                return True
        return left_reached and right_reached

    def execute_grasp(self, obj_position: np.ndarray, obj_width: float, grasp_clearance: float,
                      class_name: str, force_threshold: float = 10) -> bool:
        relay_left, relay_right, final_left, final_right = self.calculate_targets(obj_position, obj_width,
                                                                                  grasp_clearance, class_name)
        print(f"Relay targets - Left: {relay_left}, Right: {relay_right}")
        print(f"Final targets - Left: {final_left}, Right: {final_right}")

        # 移動到中繼位置
        print("Moving to relay positions")
        if not self.move_to_relay(relay_left, relay_right):
            print("Failed to reach relay positions")
            return False

        # 從中繼位置移動到最終位置，實時檢查觸碰
        print("Moving both arms from relay to final grasp positions")
        left_current = self.get_current_angles("left")
        right_current = self.get_current_angles("right")
        left_target = self.inverse_kinematics(final_left, left_current, "left")
        right_target = self.inverse_kinematics(final_right, right_current, "right")
        left_trajectory = np.linspace(left_current, left_target, self.step_size)
        right_trajectory = np.linspace(right_current, right_target, self.step_size)

        left_contact = False
        right_contact = False
        for i in range(self.step_size):
            # 若尚未觸碰，繼續移動手臂
            if not left_contact:
                self.set_angles(left_trajectory[i], "left")
            if not right_contact:
                self.set_angles(right_trajectory[i], "right")
            self.step(self.timestep)

            # 檢查左臂是否碰到物體
            if not left_contact and self.is_touch_detected("left", threshold=force_threshold):
                print(f"Left arm detected contact with force > {force_threshold} N at step {i}, stopping movement")
                self.close_fingers_custom("left", force_threshold=force_threshold)
                left_contact = True

            # 檢查右臂是否碰到物體
            if not right_contact and self.is_touch_detected("right", threshold=force_threshold):
                print(f"Right arm detected contact with force > {force_threshold} N at step {i}, stopping movement")
                self.close_fingers_custom("right", force_threshold=force_threshold)
                right_contact = True

            # 若雙手都觸碰到物體，停止移動並準備抬起
            if left_contact and right_contact:
                print("Both arms have contacted the object, stopping movement")
                break

        # 若移動完成仍未雙手觸碰，檢查最終位置是否觸碰
        if not (left_contact and right_contact):
            print(f"Reached final position without full contact: Left={left_contact}, Right={right_contact}")
            if self.is_touch_detected("left", threshold=force_threshold) and not left_contact:
                print("Left arm detected contact at final position, stopping and closing fingers")
                self.close_fingers_custom("left", force_threshold=force_threshold)
                left_contact = True
            if self.is_touch_detected("right", threshold=force_threshold) and not right_contact:
                print("Right arm detected contact at final position, stopping and closing fingers")
                self.close_fingers_custom("right", force_threshold=force_threshold)
                right_contact = True

        # 若雙手都接觸到物體，執行抬起動作
        if left_contact and right_contact:
            print("Grasp completed, lifting object")
            self.lift_object(final_left, final_right)
            return True
        else:
            print(f"Grasp failed: Left contact={left_contact}, Right contact={right_contact}")
            self.initialize_arms()
            return False

    def lift_object(self, final_left: np.ndarray, final_right: np.ndarray) -> None:
        lift_height = 0.05
        left_current = self.get_current_angles("left")
        right_current = self.get_current_angles("right")
        lift_left = np.array([final_left[0], final_left[1], final_left[2] + lift_height])
        lift_right = np.array([final_right[0], final_right[1], final_right[2] + lift_height])
        left_target = self.inverse_kinematics(lift_left, left_current, "left")
        right_target = self.inverse_kinematics(lift_right, right_current, "right")
        left_trajectory = np.linspace(left_current, left_target, self.step_size)
        right_trajectory = np.linspace(right_current, right_target, self.step_size)
        for i in range(self.step_size):
            self.set_angles(left_trajectory[i], "left")
            self.set_angles(right_trajectory[i], "right")
            self.step(self.timestep)
        print("Object lifted by 5 cm")
        
    def run(self) -> None:
        print("Starting NAO robot system for grasping with feedback control")

        # Motion distances
        sidestep_large_distance = 0.065
        sidestep_small_distance = 0.052
        forward_distance = 0.09
        forward_small_distance = 0.03

        # 目標範圍與閾值
        Y_TOLERANCE = 0.05
        X_TOLERANCE = 0.015
        Y_LOOP_THRESHOLD = 0.1
        RETREAT_STEPS = 2  # 後退步數

        # 初始化手臂
        self.initialize_arms()

        while self.step(self.timestep) != -1:
            frame = self.get_camera_image()
            if frame is None:
                continue

            processed_frame, detected_objects = self.process_frame(frame, self.get_head_angles())
            if processed_frame is None:
                continue

            cv2.imshow("NAO Camera View", processed_frame)
            if cv2.waitKey(1) == ord('q'):
                break

            if not detected_objects:
                print("No objects detected, scanning...")
                self.set_head_angles(0, -15)
                continue

            closest = min(detected_objects, key=lambda x: x['distance'])
            x_coord, y_coord, z_coord = closest['position']
            class_name = closest['class_name']
            print(f"Detected {class_name}: X={x_coord:.3f}m, Y={y_coord:.3f}m, Z={z_coord:.3f}m")

            min_x = 0.28 if class_name == "can" else 0.19
            max_x = 0.30 if class_name == "can" else 0.21
            target_x = (min_x + max_x) / 2

            if not self.align_to_target(closest['center'][0], closest['center'][1]):
                continue

            frame = self.get_camera_image()
            closest = self.update_position(frame)
            if not closest:
                continue
            x_coord, y_coord = closest['position'][0], closest['position'][1]

            # 若 X 小於最小值，後退兩步
            if x_coord < min_x:
                print(f"X={x_coord:.3f}m is less than min_x={min_x:.3f}m, retreating {RETREAT_STEPS} steps")
                self.continuous_move("Forwards", RETREAT_STEPS, reverse=True)  # 後退
                frame = self.get_camera_image()
                closest = self.update_position(frame)
                if not closest:
                    continue
                x_coord, y_coord = closest['position'][0], closest['position'][1]
                print(f"After retreat: X={x_coord:.3f}m, Y={y_coord:.3f}m")

            # 若 X 在範圍內但 Y 超出範圍，後退並重新對準
            if min_x <= x_coord <= max_x and abs(y_coord) > Y_TOLERANCE:
                print(
                    f"X in range [{min_x}, {max_x}], but Y={y_coord:.3f}m out of [-{Y_TOLERANCE}, {Y_TOLERANCE}], retreating")
                self.continuous_move("Forwards", RETREAT_STEPS, reverse=True)  # 後退
                frame = self.get_camera_image()
                closest = self.update_position(frame)
                if not closest:
                    continue
                x_coord, y_coord = closest['position'][0], closest['position'][1]
                print(f"After retreat: X={x_coord:.3f}m, Y={y_coord:.3f}m")

            # Y 軸調整（優先）
            if abs(y_coord) > Y_TOLERANCE:
                if abs(y_coord) > Y_LOOP_THRESHOLD:
                    y_diff = abs(y_coord) - Y_LOOP_THRESHOLD
                    steps = int(y_diff // sidestep_large_distance) + (1 if y_diff % sidestep_large_distance > 0 else 0)
                    direction = "SideStepLeft" if y_coord > 0 else "SideStepRight"
                    print(f"Y={y_coord:.3f}m, moving {direction} for {steps} steps to reduce to {Y_LOOP_THRESHOLD}m")
                    self.continuous_move(direction, steps)
                else:
                    while abs(y_coord) > Y_TOLERANCE:
                        motion = "SideStepLeft08" if y_coord > 0 else "SideStepRight08"
                        self.play_motion(motion)
                        frame = self.get_camera_image()
                        closest = self.update_position(frame)
                        if not closest:
                            break
                        y_coord = closest['position'][1]
                        print(f"Updated Y={y_coord:.3f}m")
                continue
            self.initialize_arms()

            # X 軸調整（Y 調整完成後）
            x_diff = x_coord - target_x
            if abs(x_diff) > X_TOLERANCE:
                if abs(x_diff) > Y_LOOP_THRESHOLD:
                    steps = int(abs(x_diff) // forward_distance) + (1 if abs(x_diff) % forward_distance > 0 else 0)
                    reverse = x_diff < 0
                    print(f"X={x_coord:.3f}m, moving Forwards (reverse={reverse}) for {steps} steps")
                    self.continuous_move("Forwards", steps, reverse=reverse)
                else:
                    while abs(x_diff) > X_TOLERANCE:
                        reverse = x_diff < 0
                        motion = "ForwardsSmall" if abs(x_diff) <= forward_distance else "Forwards"
                        self.play_motion(motion, reverse=reverse)
                        frame = self.get_camera_image()
                        closest = self.update_position(frame)
                        if not closest:
                            break
                        x_coord = closest['position'][0]
                        x_diff = x_coord - target_x
                        print(f"Updated X={x_coord:.3f}m")
                continue

            # 位置對準後執行抓取
            if (min_x <= x_coord <= max_x) and (abs(y_coord) <= Y_TOLERANCE):
                print(f"Position aligned: X={x_coord:.3f}m, Y={y_coord:.3f}m, executing grasp")
                success = self.execute_grasp(closest['position'], closest['width'], closest['grasp_clearance'],
                                             class_name)
                if success:
                    print("Successfully grasped object")
                    break
                else:
                    print("Grasp failed, reinitializing")
                    self.initialize_arms()
                    continue

        cv2.destroyAllWindows()
        print("NAO robot system stopped")

if __name__ == "__main__":
    model_path = ".../yolov8/yolov8/yaml/exp__yolov8_train11265/weights/best.pt"
    robot = Nao(model_path)
    robot.run()
