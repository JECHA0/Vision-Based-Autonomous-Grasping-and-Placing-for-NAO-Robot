from controller import Motion
import os

class MotionControl:
    def __init__(self, robot):
        self.robot = robot
        self.motions = {}
        base_path = "C:/Program Files/Webots/projects/robots/softbank/nao/motions/"
        motion_files = ["SideStepLeft.motion", "SideStepRight.motion", "SideStepLeft08.motion",
                        "SideStepRight08.motion", "Forwards.motion", "ForwardsSmall.motion", "Backwards.motion","TurnLeft180.motion"]
        for motion_file in motion_files:
            motion_name = motion_file.split('.')[0]
            motion = Motion(os.path.join(base_path, motion_file))
            if motion.isValid():
                self.motions[motion_name] = motion
                print(f"Loaded {motion_name} successfully, duration: {motion.getDuration()} ms")
            else:
                print(f"Failed to load {motion_name}, skipping")
                del motion

    def continuous_move(self, motion_name: str, steps: int, reverse: bool = False) -> bool:
        motion = self.motions.get(motion_name)
        if not motion or not motion.isValid():
            print(f"Motion {motion_name} not available or invalid")
            return False

        motion.setReverse(reverse)
        motion.setLoop(True)
        motion.play()
        print(f"Starting continuous {motion_name}, steps={steps}, reverse={reverse}")

        total_steps = steps * (motion.getDuration() // self.robot.timestep)
        for i in range(total_steps):
            self.robot.step(self.robot.timestep)
            # 修改這裡，使用 self.robot.arm.is_touch_detected
            if i % 10 == 0 and (self.robot.arm.is_touch_detected("left", 1.0) or self.robot.arm.is_touch_detected("right", 1.0)):
                motion.stop()
                print("Touch detected, stopping motion")
                self.robot.arm.initialize_moves()  # 這裡也更新為 arm.initialize_moves
                return False
        motion.stop()
        print(f"Completed {motion_name} for {steps} steps")
        return True

    def play_motion(self, motion_name: str, reverse: bool = False) -> bool:
        motion = self.motions.get(motion_name)
        if not motion or not motion.isValid():
            print(f"Motion {motion_name} not available or invalid")
            return False
        motion.setReverse(reverse)
        motion.play()
        while not motion.isOver():
            self.robot.step(self.robot.timestep)
        print(f"Completed {motion_name}, reverse={reverse}")
        return True