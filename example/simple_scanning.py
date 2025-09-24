from controller import Robot
import numpy as np
import cv2
import os
from ultralytics import YOLO
from vision_system import VisionSystem

class Nao(Robot):
    def __init__(self, model_path: str):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        # Camera initialization
        self.camera = self.getDevice('CameraBottom')
        self.camera.enable(self.timestep)
        self.image_width = self.camera.getWidth()
        self.image_height = self.camera.getHeight()

        # Head motors and sensors
        self.head_yaw_sensor = self.getDevice('HeadYawS')
        self.head_pitch_sensor = self.getDevice('HeadPitchS')
        self.head_yaw_sensor.enable(self.timestep)
        self.head_pitch_sensor.enable(self.timestep)
        self.head_yaw_motor = self.getDevice('HeadYaw')
        self.head_pitch_motor = self.getDevice('HeadPitch')

        # Object reference data
        self.object_reference = {
            'can': {'height': 0.1222, 'diameter': 0.03175, 'grasp_clearance': 0.04},
            'ball': {'diameter': 0.0325, 'grasp_clearance': 0.03},
            'cookie box': {'height': 0.005, 'diameter': 0.03175, 'grasp_clearance': 0.04},
            'duck': {'diameter': 0.01066, 'grasp_clearance': 0.03},
            'screw': {'diameter': 0.0, 'grasp_clearance': 0.03}
        }
        self.camera_x = 0.05071
        self.camera_z = 0.01774
        self.torso_z_offset = 0.1265

        # YOLO model
        self.yolo_model = YOLO(model_path) if os.path.exists(model_path) else None
        if self.yolo_model is None:
            raise ValueError(f"YOLO model not found at {model_path}")

        # Head controller limits
        self.yaw_limits = (-2.09, 2.09)  # radians
        self.pitch_limits = (-0.67, 0.51)  # radians

        # Initialize vision system
        self.vision = VisionSystem(self)

    def _scan_for_target(self, yaw_range, yaw_speed):
        """Continuously scan environment left-right with ultra-smooth head movement."""
        print("Entering head search phase")
        current_yaw = 0  # Start at Yaw=0°
        fixed_pitch = -15  # Fixed Pitch at -15°
        yaw_direction = 1  # Initially scan right

        # Convert yaw range to radians for internal calculations
        yaw_range_rad = (np.deg2rad(yaw_range[0]), np.deg2rad(yaw_range[1]))
        yaw_speed_rad = np.deg2rad(yaw_speed)

        self.vision.set_head_angles(current_yaw, fixed_pitch)

        while self.step(self.timestep) != -1:
            image = self.vision.get_camera_image()
            if image is None:
                continue
            head_angles = self.vision.get_head_angles()
            processed_frame, detected_objects = self.vision.process_frame(image, head_angles)
            if processed_frame is not None:
                cv2.imshow("NAO Camera View", processed_frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    raise SystemExit("User terminated the program")
            if detected_objects:
                target_obj = min(detected_objects, key=lambda x: abs(x['position'][1]))
                print(f"Object found: {target_obj['class_name']} at {target_obj['position']}")

            # Update yaw angle with smaller increments for smoother movement
            delta_yaw = yaw_speed_rad * yaw_direction * (self.timestep / 1000.0)  # Convert timestep to seconds
            current_yaw += delta_yaw

            # Ensure yaw stays within limits
            if current_yaw >= yaw_range_rad[1]:
                current_yaw = yaw_range_rad[1]
                yaw_direction = -1
            elif current_yaw <= yaw_range_rad[0]:
                current_yaw = yaw_range_rad[0]
                yaw_direction = 1

            # Apply smoothing using interpolation
            smoothed_yaw = np.clip(current_yaw, yaw_range_rad[0], yaw_range_rad[1])
            print(f"Scanning: yaw={np.rad2deg(smoothed_yaw):.1f}°, pitch={fixed_pitch:.1f}°")
            self.vision.set_head_angles(np.rad2deg(smoothed_yaw), fixed_pitch)

    def run(self) -> None:
        """Main loop to control NAO robot for continuous left-right scanning."""
        print("Starting NAO robot system for scanning")

        # Head scanning parameters
        YAW_RANGE = (-60, 60)  # degrees
        YAW_SPEED = 90  # degrees per second (fast but smooth)

        # Initialize head position
        self.vision.set_head_angles(0, -15)
        for _ in range(50):
            self.step(self.timestep)

        # Start continuous left-right scanning
        self._scan_for_target(YAW_RANGE, YAW_SPEED)

        cv2.destroyAllWindows()
        print("NAO robot scanning stopped")

if __name__ == "__main__":
    model_path = "F:/專題資料紀錄/12042/my_project123_1125/yolov8/yolov8/yaml/exp__yolov8_train11265/weights/best.pt"
    robot = Nao(model_path)
    robot.run()