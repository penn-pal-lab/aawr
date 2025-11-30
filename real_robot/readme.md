# Real Robot Code
Jie Wang
11/30/2025

> Note: the code is under construction. 
> Please contact the authors or Github issues if you have any questions.

To operate on the real robot, we provides two platforms:

1. Franka EVA
   - We use the Franka EVA framework to operate the DROID setup.
   - It supports the following features:
     - Teleoperation using Oculus, Keyboard, or SpaceMouse
     - Policy Inference
     - Trajectory Collection
     - Trajectory Playback
     - Trajectory Processing
     - Camera Calibration
     - Robot Reset
   - For more details, please refer to the [Franka EVA README](franka_eva/README.md).

2. Koch LeRobot
   - We use the LeRobot framework to operate the Koch 1.0 robot.
   - It supports the following features:
     - Teleoperation using Leader Arm, Game Controller, or Keyboard
     - Policy Inference
     - Trajectory Collection and Playback
     - Trajectory Processing
     - Camera Calibration
     - Robot Reset
   - For more details, please refer to the [Koch LeRobot README](koch_lerobot/README.md).