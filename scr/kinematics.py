import numpy as np

class Kinematics:
    def __init__(self, robot):
        self.robot = robot
        self.L = robot.L
        self.joint_limits = robot.joint_limits

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
        min_angle, max_angle = self.joint_limits.get(joint_name, self.robot.phalanx_limits)
        return max(min(angle, max_angle), min_angle)