from controller import Supervisor  # Change this import
import numpy as np
import cv2
import os
from ultralytics import YOLO
from motion_control import MotionControl
from vision_system import VisionSystem
from kinematics import Kinematics
from arm_control import ArmControl
from grasp_planner import GraspPlanner

class Nao(Supervisor):  # Inherit from Supervisor instead of Robot
    def __init__(self, model_path: str):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        self.step_size = 20

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
            sensor.enable(self.timestep)
        self.left_phalanxes = {f"LPhalanx{i}": self.getDevice(f"LPhalanx{i}") for i in range(1, 9)}
        self.right_phalanxes = {f"RPhalanx{i}": self.getDevice(f"RPhalanx{i}") for i in range(1, 9)}

        # Touch sensors
        self.left_touch_sensor = self.getDevice('Ltouchsensor')
        self.right_touch_sensor = self.getDevice('Rtouchsensor')
        self.left_touch_sensor.enable(self.timestep)
        self.right_touch_sensor.enable(self.timestep)

        # Object reference data
        self.object_reference = {
            'can': {'height': 0.1222, 'diameter': 0.03175, 'grasp_clearance': 0.06},
            'ball': {'diameter': 0.0325, 'grasp_clearance': 0.03}
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

        # Kinematics parameters
        self.L = [0.098, 0.100, 0.015, 0.105, 0.05595, 0.05775, 0.01231]
        self.joint_limits = {
            "LShoulderPitch": (-2.0857, 2.0857), "LShoulderRoll": (-0.314159, 1.3265),
            "LElbowYaw": (-2.0857, 2.0857), "LElbowRoll": (-1.5446, -0.0349),
            "LWristYaw": (-1.8239, 1.8239), "RShoulderPitch": (-2.0857, 2.0857),
            "RShoulderRoll": (-1.3265, 0.314159), "RElbowYaw": (-2.0857, 2.0857),
            "RElbowRoll": (0.0349, 1.5446), "RWristYaw": (-1.8239, 1.8239)
        }
        self.phalanx_limits = (0.0, 1.0)

        # Initialize modular components
        self.motion = MotionControl(self)
        self.vision = VisionSystem(self)
        self.kinematics = Kinematics(self)
        self.arm = ArmControl(self)
        self.grasp_planner = GraspPlanner(self)

        #stroage box world position
        self.storage_boxes={
            'Screw Storage box': [-0.65, 2, 0],
            'Cookiebox Storage box':  [-0.65, 1, 0],
            'Can Storage box': [-0.65, -2 , 0],
            'Ball Storage box': [-0.65, 0, 0 ],
            'Duck Storage box': [-0.65, -1, 0]
        }
    """0422 new add NAO translation in world coordinary"""
    def get_nao_world_position(self):
        nao_node = self.getFromDef("Nao")
        nao_pos = nao_node.getField("translation").getSFVec3f()    #這裡是得到NAO在世界座標的translation(絕對座標)
        return nao_pos

    def run(self) -> None:
        """Main loop to control NAO robot for object grasping."""
        print("Starting NAO robot system for object grasping")

        # Motion distances (in meters)
        SIDESTEP_LARGE = 0.065
        SIDESTEP_SMALL = 0.052
        FORWARD_LARGE = 0.09
        FORWARD_SMALL = 0.03

        # Tolerances and thresholds
        Y_TOLERANCE = 0.03
        X_TOLERANCE = 0.015
        Y_THRESHOLD = 0.1
        RETREAT_STEPS = 2
        MAX_ATTEMPTS = 5
        MAX_ALIGNMENT_ATTEMPTS = 5

        # Head scanning parameters
        YAW_RANGE = (-60, 60)  # degrees
        PITCH_RANGE = (-25, -15)  # degrees
        YAW_STEP = 20
        PITCH_STEP = 10

        # Initialize robot
        self.arm.initialize_moves()
        self.vision.set_head_angles(0, 0)
        for _ in range(50):
            self.step(self.timestep)

        # Main loop
        while self.step(self.timestep) != -1:
            # Step 1: Scan for target
            target_obj = self._scan_for_target(YAW_RANGE, PITCH_RANGE, YAW_STEP, PITCH_STEP)
            if not target_obj:
                print("No objects found, restarting scan")
                continue

            # Step 2: Align and adjust position
            aligned_target = self._align_and_adjust(
                target_obj, SIDESTEP_LARGE, SIDESTEP_SMALL, FORWARD_LARGE, FORWARD_SMALL,
                Y_TOLERANCE, X_TOLERANCE, Y_THRESHOLD, RETREAT_STEPS, MAX_ATTEMPTS, MAX_ALIGNMENT_ATTEMPTS
            )
            if not aligned_target:
                print("Alignment failed or target lost, restarting scan")
                continue

            # Step 3: Attempt grasp
            grasped = self._attempt_grasp(aligned_target)
            if grasped:
                print("Grasp successful, stopping system")
                cv2.destroyAllWindows()
                return
            else:
                print("Grasp failed, reinitializing and retrying")
                self.arm.initialize_moves()

        cv2.destroyAllWindows()
        print("NAO robot system stopped")

    def _scan_for_target(self, yaw_range, pitch_range, yaw_step, pitch_step):
        """Scan environment to detect target object, starting from Yaw=0, Pitch=-15."""
        print("Entering head search phase")
        current_yaw = 0  # Start at Yaw=0°
        current_pitch = -15  # Start at Pitch=-15°
        yaw_direction = 1  # Initially scan right

        self.vision.set_head_angles(current_yaw, current_pitch)
        for _ in range(10):  # Stabilize head
            self.step(self.timestep)

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
                self.vision.set_head_angles(np.degrees(head_angles[0]), np.degrees(head_angles[1]))
                for _ in range(20):
                    self.step(self.timestep)
                return target_obj

            # Update head angles
            current_pitch += pitch_step
            if current_pitch > pitch_range[1]:
                current_pitch = pitch_range[0]
                current_yaw += yaw_step * yaw_direction
                if current_yaw >= yaw_range[1]:
                    current_yaw = yaw_range[1]
                    yaw_direction = -1
                elif current_yaw <= yaw_range[0]:
                    current_yaw = yaw_range[0]
                    yaw_direction = 1
            print(f"Scanning: yaw={current_yaw:.1f}°, pitch={current_pitch:.1f}°")
            self.vision.set_head_angles(current_yaw, current_pitch)
            for _ in range(10):
                self.step(self.timestep)
        return None

    def _align_and_adjust(self, target_obj, sidestep_large, sidestep_small, forward_large, forward_small,
                          y_tolerance, x_tolerance, y_threshold, retreat_steps, max_attempts, max_alignment_attempts):
        """Align robot with target and adjust X and Y position, return final target object."""
        print("Entering alignment and adjustment phase")
        attempts = 0

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

            if not detected_objects:
                attempts += 1
                print(f"No detection, attempt {attempts}/{max_attempts}")
                if attempts >= max_attempts:
                    print("Target lost, resuming scan")
                    return None
                continue

            attempts = 0
            target_obj = min(detected_objects, key=lambda x: abs(x['position'][1]))
            x_coord, y_coord = target_obj['position'][0], target_obj['position'][1]
            class_name = target_obj['class_name']
            min_x, max_x = (0.285, 0.325) if class_name == "can" else (0.19, 0.21)
            print(f"Current position: X={x_coord:.3f}m, Y={y_coord:.3f}m")

            # Y-axis adjustment (priority)
            if abs(y_coord) > y_tolerance:
                if abs(y_coord) > y_threshold:
                    y_diff = abs(y_coord) - y_threshold
                    steps = int(y_diff // sidestep_large) + (1 if y_diff % sidestep_large > 0 else 0)
                    direction = "SideStepLeft" if y_coord > 0 else "SideStepRight"
                    print(f"Y={y_coord:.3f}m, moving {direction} for {steps} steps")
                    self.motion.continuous_move(direction, steps)
                else:
                    while abs(y_coord) > y_tolerance:
                        motion = "SideStepLeft08" if y_coord > 0 else "SideStepRight08"
                        print(f"Y={y_coord:.3f}m, fine-tuning with {motion}")
                        self.motion.play_motion(motion)
                        image = self.vision.get_camera_image()
                        if image is not None:
                            updated_target = self.vision.update_position(image)
                            if updated_target:
                                target_obj = updated_target
                                y_coord = target_obj['position'][1]
                                print(f"Updated Y={y_coord:.3f}m")
                            else:
                                print("Target lost during Y adjustment")
                                return None
                        else:
                            print("Failed to capture image during Y adjustment")
                            return None
                continue

            # Initialize arms before X-axis adjustment
            self.arm.initialize_moves()

            # X-axis adjustment
            if x_coord < min_x:
                print(f"X={x_coord:.3f}m < {min_x:.3f}m, retreating {retreat_steps} steps")
                self.motion.continuous_move("Forwards", retreat_steps, reverse=True)
                continue
            elif x_coord >= 0.4:
                steps = int((x_coord - max_x) / forward_large)
                print(f"X={x_coord:.3f}m >= 0.4m, moving forward {steps} steps")
                self.motion.continuous_move("Forwards", steps)
                continue
            elif max_x < x_coord < 0.4:
                while x_coord > max_x:
                    print(f"X={x_coord:.3f}m between {max_x:.3f}m and 0.4m, small step forward")
                    self.motion.play_motion("ForwardsSmall")
                    image = self.vision.get_camera_image()
                    if image is not None:
                        updated_target = self.vision.update_position(image)
                        if updated_target:
                            target_obj = updated_target
                            x_coord = target_obj['position'][0]
                            print(f"Updated X={x_coord:.3f}m")
                        else:
                            print("Target lost during X adjustment")
                            return None
                    else:
                        print("Failed to capture image during X adjustment")
                        return None
                continue

            # Final check
            if min_x <= x_coord <= max_x and abs(y_coord) <= y_tolerance:
                print(f"Aligned: X={x_coord:.3f}m, Y={y_coord:.3f}m")
                return target_obj

        return None

    def _attempt_grasp(self, target_obj):
        """Attempt to grasp the target object with both arms."""
        #print(f"Aligned: X={target_obj['position'][0]:.3f}m, Y={target_obj['position'][1]:.3f}m, attempting grasp")
        success = self.grasp_planner.execute_grasp(
            target_obj['position'], target_obj['width'], target_obj['grasp_clearance'],
            target_obj['class_name'], force_threshold=1.8 #threshold太大
        )
        if success:
            self.motion.continuous_move("Forwards", 1, reverse=True) #step 1 先後退一步(inverse forwards)
            position_before = self.get_nao_world_position() #旋轉前的nao translation
            self.motion.play_motion("TurnLeft180") #step 2 旋轉180度 向後轉
            position_after = self.get_nao_world_position() #旋轉後的nao translation

            #從抓到物體的object_name來對應box位置
            class_name = target_obj['class_name'].lower() # class name 和box name# 不一樣
            box_mapping = {
                'screw': 'Screw Storage box',
                'cookiebox': 'Cookiebox Storage box',
                'can': 'Can Storage box',
                'ball': 'Ball Storage box',
                'duck': 'Duck Storage box'
            }
            target_box_name = box_mapping.get(class_name)
            target_box_position = self.storage_boxes[target_box_name]
            #移動到storage box
            self._place_object(target_box_position)

    def _place_object(self, target_position):
        """移動到指定物體放置位置和放置物體(鬆開手指)，X range: 0.25~0.27 m，Y range:-0.02~0.02 m"""
        X_DISTANCE_MIN = 0.25  # 最小X距離
        X_DISTANCE_MAX = 0.27  # 最大X距離
        Y_TOLERANCE = 0.02  # Y 方向可放置範圍（-0.02 ~ 0.02）

        target_x, target_y, target_z = target_position

        while self.step(self.timestep) != -1:
            # NAO的世界座標（每次都更新NAO translation）
            current_position = self.get_nao_world_position()
            current_x, current_y, current_z = current_position
            print(f"Current world position: X={current_x:.3f}, Y={current_y:.3f}, Z={current_z:.3f}")

            # 計算 X 和 Y 方向的距離（NAO 座標-物體放置座標）
            delta_x = current_x - target_x
            delta_y = current_y - target_y
            print(f"Distance to target: delta_x={delta_x:.3f}m, delta_y={delta_y:.3f}m")

            # Y 方向對準（世界座標系）
            if abs(delta_y) > Y_TOLERANCE:
                # 以delta_y決定左右側步方向
                if delta_y > 0:
                    # NAO 的 Y 座標>目標，需要向左側移
                    if abs(delta_y) > 0.1:  # 用大步伐
                        steps = int(abs(delta_y) // 0.065) + (1 if abs(delta_y) % 0.065 > 0 else 0)
                        print(f"Y={current_y:.3f}m, moving SideStepLeft for {steps} steps")
                        self.motion.continuous_move("SideStepLeft", steps)
                    else:
                        print(f"Y={current_y:.3f}m, fine-tuning with SideStepLeft08")
                        self.motion.play_motion("SideStepLeft08")
                else:
                    # NAO 的 Y 座標<目標，向右側移
                    if abs(delta_y) > 0.1:  # 用大步伐
                        steps = int(abs(delta_y) // 0.065) + (1 if abs(delta_y) % 0.065 > 0 else 0)
                        print(f"Y={current_y:.3f}m, moving SideStepRight for {steps} steps")
                        self.motion.continuous_move("SideStepRight", steps)
                    else:
                        print(f"Y={current_y:.3f}m, fine-tuning with SideStepRight08")
                        self.motion.play_motion("SideStepRight08")
                continue

            # X 方向對準
            # 計算NAO translation與指定物體放置位置的中心-> X距離
            distance_to_box = abs(delta_x)  # 直接用delta_x的絕對值為距離

            print(f"Distance to box (X={target_x:.3f}): {distance_to_box:.3f}m")

            # distance_to_box範圍調整NAO的前進步伐
            if distance_to_box < X_DISTANCE_MIN:
                # 太近->後退
                distance_to_adjust = X_DISTANCE_MIN - distance_to_box
                if distance_to_adjust > 0.09:
                    steps = int(distance_to_adjust // 0.09) + (1 if distance_to_adjust % 0.09 > 0 else 0)
                    print(f"Too close ({distance_to_box:.3f}m < {X_DISTANCE_MIN:.3f}m), retreating {steps} steps")
                    self.motion.continuous_move("Forwards", steps, reverse=True)
                else:
                    print(f"Too close ({distance_to_box:.3f}m < {X_DISTANCE_MIN:.3f}m), fine-tuning retreat")
                    self.motion.play_motion("ForwardsSmall", reverse=True)
            elif distance_to_box > X_DISTANCE_MAX:
                # 太遠->以delta_x決定移動方向
                if delta_x > 0:
                    # NAO 在BOX右側（delta_x > 0），就前進（世界座標系 -> -X方向）
                    distance_to_adjust = distance_to_box - X_DISTANCE_MAX
                    if distance_to_adjust > 0.09:
                        steps = int(distance_to_adjust // 0.09) + (1 if distance_to_adjust % 0.09 > 0 else 0)
                        print(f"Too far ({distance_to_box:.3f}m > {X_DISTANCE_MAX:.3f}m), advancing {steps} steps")
                        self.motion.continuous_move("Forwards", steps, reverse=False)
                    else:
                        print(f"Too far ({distance_to_box:.3f}m > {X_DISTANCE_MAX:.3f}m), fine-tuning advance")
                        self.motion.play_motion("ForwardsSmall", reverse=False)
                else:
                    # NAO 在BOX左側（delta_x < 0），就後退（世界座標系 -> X方向）
                    distance_to_adjust = distance_to_box - X_DISTANCE_MAX
                    if distance_to_adjust > 0.09:
                        steps = int(distance_to_adjust // 0.09) + (1 if distance_to_adjust % 0.09 > 0 else 0)
                        print(
                            f"Too far on left side ({distance_to_box:.3f}m > {X_DISTANCE_MAX:.3f}m), retreating {steps} steps")
                        self.motion.continuous_move("Forwards", steps, reverse=True)
                    else:
                        print(
                            f"Too far on left side ({distance_to_box:.3f}m > {X_DISTANCE_MAX:.3f}m), fine-tuning retreat")
                        self.motion.play_motion("ForwardsSmall", reverse=True)
            else:
                # 距離在範圍內，且 Y 方向已對準
                if abs(delta_y) <= Y_TOLERANCE:
                    print(
                        f"Reached target position: X={current_x:.3f}, Y={current_y:.3f}, Distance to box: {distance_to_box:.3f}m")
                    self.arm.release_object()
                    break
                continue
if __name__ == "__main__":
    model_path = "F:/1132機器學習/train5/weights/best.pt"
    robot = Nao(model_path)
    robot.run()