import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict

class VisionSystem:
    def __init__(self, robot):
        self.robot = robot
        self.yolo_model = robot.yolo_model
        self.image_width = robot.image_width
        self.image_height = robot.image_height
        self.hfov = 60.97
        self.vfov = 47.64
        self.horizontal_focal_length = (self.image_width / 2) / np.tan(np.radians(self.hfov / 2))
        self.vertical_focal_length = (self.image_height / 2) / np.tan(np.radians(self.vfov / 2))
        self.object_reference = robot.object_reference
        self.camera_x = robot.camera_x
        self.camera_z = robot.camera_z
        self.torso_z_offset = robot.torso_z_offset
        self.yaw_limits = robot.yaw_limits
        self.pitch_limits = robot.pitch_limits

    def get_camera_image(self) -> Optional[np.ndarray]:
        image = self.robot.camera.getImage()
        if image:
            return cv2.cvtColor(np.frombuffer(image, np.uint8).reshape((self.image_height, self.image_width, 4)),
                                cv2.COLOR_RGBA2BGR)
        return None

    def get_head_angles(self) -> Tuple[float, float]:
        return self.robot.head_yaw_sensor.getValue(), self.robot.head_pitch_sensor.getValue()

    def process_frame(self, frame: np.ndarray, head_angles: Tuple[float, float]) -> Tuple[Optional[np.ndarray], List[Dict]]:
        if frame is None or self.yolo_model is None:
            return None, []
        frame_resized = cv2.resize(frame, (self.image_width, self.image_height))
        results = self.yolo_model(frame_resized, conf=0.93)[0]
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
            position[2] -= self.object_reference[class_name]['height'] * 0.5
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

    def set_head_angles(self, yaw_deg: float, pitch_deg: float) -> None:
        yaw_rad = np.radians(np.clip(yaw_deg, -119.5, 119.5))
        pitch_rad = np.radians(np.clip(pitch_deg, -29.5, 38.5))
        if self.yaw_limits[0] <= yaw_rad <= self.yaw_limits[1] and self.pitch_limits[0] <= pitch_rad <= self.pitch_limits[1]:
            self.robot.head_yaw_motor.setPosition(yaw_rad)
            self.robot.head_pitch_motor.setPosition(pitch_rad)

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

    def update_position(self, frame):
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