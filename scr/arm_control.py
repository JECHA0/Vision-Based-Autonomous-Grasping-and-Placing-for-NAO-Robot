import numpy as np

class ArmControl:
    def __init__(self, robot):
        self.robot = robot
        self.left_arm_motors = robot.left_arm_motors
        self.right_arm_motors = robot.right_arm_motors
        self.left_arm_sensors = robot.left_arm_sensors
        self.right_arm_sensors = robot.right_arm_sensors
        self.left_phalanxes = robot.left_phalanxes
        self.right_phalanxes = robot.right_phalanxes
        self.kinematics = robot.kinematics

    def is_touch_detected(self, side: str, threshold: float = 1) -> bool:
        sensor = self.robot.left_touch_sensor if side == "left" else self.robot.right_touch_sensor
        force_value = sensor.getValue()
        print(f"{side} touch sensor force value: {force_value:.3f} N")
        return force_value > threshold

    def set_angles(self, angles: np.ndarray, arm_side: str) -> None:
        motors = self.left_arm_motors if arm_side == "left" else self.right_arm_motors
        joint_names = list(motors.keys())
        clamped_angles = [self.kinematics.clamp_angle(angle, name) for angle, name in zip(angles, joint_names)]
        for angle, name in zip(clamped_angles, joint_names):
            motors[name].setPosition(angle)

    def get_current_angles(self, arm_side: str) -> np.ndarray:
        sensors = self.left_arm_sensors if arm_side == "left" else self.right_arm_sensors
        return np.array([sensor.getValue() for sensor in sensors.values()])

    def move_to(self, target_pos: np.ndarray, arm_side: str, steps: int = 40) -> None:
        current_angles = self.get_current_angles(arm_side)
        target_angles = self.kinematics.inverse_kinematics(target_pos, current_angles, arm_side)
        trajectory = np.linspace(current_angles, target_angles, steps)
        for angles in trajectory:
            self.set_angles(angles, arm_side)
            self.robot.step(self.robot.timestep)

    def open_fingers(self, arm_side: str) -> None:
        phalanxes = self.left_phalanxes if arm_side == "left" else self.right_phalanxes
        for phalanx in phalanxes.values():
            phalanx.setPosition(self.robot.phalanx_limits[1])
        print(f"{arm_side} hand fingers opened")

    def close_fingers_custom(self, arm_side: str, target_position: float = 0.7, steps: int = 40,
                             force_threshold: float = 10) -> None:
        phalanxes = self.left_phalanxes if arm_side == "left" else self.right_phalanxes
        start_pos = self.robot.phalanx_limits[1]
        step_size = (start_pos - target_position) / steps
        finger_groups = [[1, 2, 3], [4, 5, 6], [7, 8]]

        print(f"{arm_side} hand: Starting finger closing to position {target_position}")

        for step in range(steps + 1):
            if self.is_touch_detected(arm_side, threshold=force_threshold):
                print(f"{arm_side} hand: Force detected above {force_threshold} N, stopping arm movement and closing fingers")
                break

            current_pos = start_pos - step * step_size
            for group in finger_groups:
                for i in group:
                    phalanx_key = f"{'L' if arm_side == 'left' else 'R'}Phalanx{i}"
                    phalanxes[phalanx_key].setPosition(current_pos)
            self.robot.step(self.robot.timestep)

        print(f"{arm_side} hand: Closing fingers to final position {target_position}")
        for group in finger_groups:
            for i in group:
                phalanx_key = f"{'L' if arm_side == 'left' else 'R'}Phalanx{i}"
                phalanxes[phalanx_key].setPosition(target_position)

        for _ in range(10):
            self.robot.step(self.robot.timestep)

        print(f"{arm_side} hand: Fingers closed to position {target_position}")

    def initialize_moves(self) -> None:
        print("Initializing both arms simultaneously")
        left_initial_angles = np.array([1.2, 0.5, -0.9, -1.54462, -0.5])
        right_initial_angles = np.array([1.2, -0.5, 0.9, 1.54462, 0.5])
        self.set_angles(left_initial_angles, "left")
        self.set_angles(right_initial_angles, "right")
        self.open_fingers("left")
        self.open_fingers("right")
        for _ in range(100):
            self.robot.step(self.robot.timestep)
        print("Both arms initialized")