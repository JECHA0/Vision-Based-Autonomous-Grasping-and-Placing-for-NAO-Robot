# NAO Robot Vision-Based Grasping and Placing System

This repository contains the source code for a comprehensive autonomous robotics project developed for the NAO robot. The system leverages computer vision with YOLO for real-time object detection and advanced kinematics to perform precise grasping and placing tasks in a simulated Webots environment.

## Core Features

* **Modular Architecture**: The code is organized into distinct modules for vision, kinematics, arm control, motion, and grasp planning, making it easy to understand and maintain.
* **Real-time Object Detection**: Utilizes a YOLO model to detect and localize objects from the NAO's camera feed.
* **3D Coordinate Estimation**: Calculates the 3D position of objects relative to the robot's torso using pinhole camera geometry and coordinate transformations.
* **Advanced Arm Control**: Implements both forward and inverse kinematics for precise 5-DOF arm control. It includes touch sensor feedback for adaptive gripping.
* **Autonomous Navigation & Alignment**: The robot can scan for objects, align its position (X and Y axes) relative to a target, and navigate to predefined locations.
* **End-to-End Task Execution**: The system can perform a full cycle: scan, detect, align, grasp, lift, turn, navigate to a drop-off zone, and release the object.

## Repository Structure