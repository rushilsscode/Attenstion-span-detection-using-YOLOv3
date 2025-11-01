# Attenstion-span-detection-using-YOLOv3
A real-time computer vision system designed to monitor and estimate a student's attention level during online classes. By leveraging YOLOv3 for person detection, MediaPipe for precise facial and hand landmarks, and OpenCV for processing, the model calculates an Attention Score based on head pose, eye gaze, and hand movements.

🧠 Project Overview
In the era of remote learning, maintaining student engagement is a significant challenge. This project provides an automated, non-intrusive tool to analyze student behavior and provide quantifiable feedback on their attentiveness.

The system processes a webcam feed to:

Detect the person in the frame using YOLOv3.

Extract key landmarks from the face, eyes, and hands using MediaPipe.

Calculate an Attention Score by analyzing:

Head Pose: Estimating the direction the head is facing (e.g., looking away from the screen).

Eye Gaze: Tracking the direction of the gaze to see if the student is looking at the screen.

Hand Movements: Detecting frequent or fidgety hand movements near the face, which can indicate distraction.

Output a real-time score and visual feedback on the video stream.

✨ Key Features
Real-time Analysis: Processes webcam feed live with low latency.

Multi-Metric Evaluation: Combines three distinct behavioral cues (Head, Eyes, Hands) for a robust attention score.

Non-Intrusive: Uses a standard webcam, requiring no specialized hardware.

Visual Feedback: Displays bounding boxes, facial landmarks, and a live attention score/meter on the screen.

Modular Design: Easily extendable to add new metrics or modify the scoring algorithm.

🛠️ Technical Architecture
Pipeline Workflow
Input: Live video stream from a webcam.

Person Detection: YOLOv3 model identifies and localizes the person in the frame.

Landmark Extraction: MediaPipe's Face Mesh and Pose models extract 468 facial landmarks and 33 pose landmarks.

Feature Calculation:

Head Pose: Calculated using the 3D-2D point correspondence with solvePnP (OpenCV).

Eye Gaze: Estimated by analyzing the relative position of the iris within the eye landmarks.

Hand Movement: Tracked by calculating the displacement of hand landmarks between consecutive frames.

Attention Scoring: A weighted algorithm combines the deviations in head pose, eye gaze, and hand movement into a single, normalized attention score (0-100%).

Output: Annotated video stream with the calculated score.

Models & Libraries Used
YOLOv3: For fast and accurate person detection.

MediaPipe: For high-performance, on-device face, eye, and hand landmark detection.

OpenCV: For all image processing, video I/O, and camera calibration tasks.

NumPy: For efficient numerical computations.

