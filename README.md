# Multi-Camera Motion Capturing and Kinematic Analysis Application

A research-oriented computer vision project developed during an AI & Robotics internship at NamTech to investigate multi-camera motion capture, 3D reconstruction, and kinematic analysis using synchronized RGB cameras, an Intel RealSense depth camera, OpenCV, and MediaPipe.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)

---

# Abstract

Human motion capture plays a significant role in robotics, biomechanics, sports analytics, healthcare, animation, and human-computer interaction. Conventional monocular vision systems often suffer from depth ambiguity, viewpoint dependency, and occlusion, limiting their ability to reconstruct accurate three-dimensional human motion.

This project investigates a low-cost multi-camera motion capture system that combines four synchronized Logitech C920 RGB cameras with an Intel RealSense depth camera to improve motion tracking accuracy. The implementation integrates camera calibration, markerless pose estimation, multi-view geometry, triangulation, and kinematic analysis to reconstruct three-dimensional joint motion from multiple viewpoints.

The project was developed as part of an AI & Robotics internship at NamTech and focuses on applying computer vision techniques to scientific and engineering problems rather than building a commercial software application.
---

# Motivation

Markerless human motion capture is increasingly used in robotics, biomechanics, healthcare, animation, and sports science. While single-camera systems are inexpensive and easy to deploy, they exhibit several fundamental limitations:

- Inability to accurately estimate depth from a single viewpoint.
- Loss of tracking during self-occlusion or object occlusion.
- Reduced accuracy caused by viewpoint dependency.
- Limited robustness for three-dimensional motion reconstruction.

This project investigates a multi-camera approach to overcome these limitations by combining observations from multiple synchronized viewpoints. Using four calibrated RGB cameras together with an Intel RealSense depth camera, the system estimates human pose from multiple perspectives and reconstructs three-dimensional joint positions for kinematic analysis.

The objective was not only to implement a functional motion capture system, but also to gain practical experience with camera calibration, multi-view geometry, pose estimation, triangulation, and scientific software development within the context of computer vision and robotics.
---

# System Overview

The Multi-Camera Motion Capturing and Kinematic Analysis Application was developed to investigate robust human motion tracking using multiple synchronized camera viewpoints. The system integrates computer vision, camera calibration, pose estimation, and three-dimensional reconstruction into a unified processing pipeline.

## Hardware

- **4 × Logitech C920 HD Pro RGB Cameras**
  - Resolution: 640 × 480
  - Frame Rate: 30 FPS
  - Used for synchronized multi-view image acquisition.

- **1 × Intel RealSense Depth Camera**
  - RGB + Depth sensing
  - Used for comparative evaluation and depth-based analysis.

## Software Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Intel RealSense SDK
- Git

## Core Components

- Multi-camera image acquisition
- Intrinsic camera calibration
- Extrinsic camera calibration
- Image undistortion
- Markerless human pose estimation
- LED-based object tracking
- Multi-view triangulation
- Three-dimensional motion reconstruction
- Kinematic analysis
---

# Computer Vision Pipeline

The motion capture system follows a sequential computer vision pipeline in which the output of each stage serves as the input for the next processing step.

```text
          Multi-Camera Image Acquisition
                     │
                     ▼
      Intrinsic Camera Calibration
                     │
                     ▼
      Extrinsic Camera Calibration
                     │
                     ▼
          Image Undistortion
                     │
                     ▼
      Markerless Pose Estimation
          (MediaPipe Pose)
                     │
                     ▼
      Multi-View Feature Association
                     │
                     ▼
        3D Triangulation (DLT)
                     │
                     ▼
      Three-Dimensional Reconstruction
                     │
                     ▼
         Kinematic Motion Analysis
```

Each stage was implemented and validated independently before integration into the complete motion capture pipeline. The modular design allows individual components, such as calibration or pose estimation, to be improved without affecting the overall architecture.
---

# Camera Calibration

Accurate camera calibration is fundamental to reliable three-dimensional reconstruction. Since triangulation relies on precise geometric relationships between cameras, even small calibration errors can significantly affect the reconstructed 3D coordinates.

The calibration process consisted of both intrinsic and extrinsic calibration using a checkerboard calibration pattern.

## Intrinsic Calibration

Intrinsic calibration was performed independently for each camera to estimate its internal imaging parameters, including:

- Camera matrix
- Focal length
- Principal point
- Lens distortion coefficients

These parameters were used to remove radial and tangential lens distortion before further image processing.

## Extrinsic Calibration

Extrinsic calibration estimated the spatial relationship between cameras by determining their rotation and translation with respect to a common world coordinate system.

The resulting transformation matrices enabled observations from multiple viewpoints to be projected into a shared three-dimensional coordinate frame.

## Calibration Workflow

```text
Checkerboard Image Capture
            │
            ▼
Corner Detection (OpenCV)
            │
            ▼
Intrinsic Parameter Estimation
            │
            ▼
Extrinsic Parameter Estimation
            │
            ▼
Image Undistortion
            │
            ▼
Multi-Camera Coordinate Alignment
```

Calibration quality was verified by evaluating detected checkerboard corners, reviewing reprojection consistency across viewpoints, and iteratively refining the calibration setup to improve overall reconstruction accuracy.
---

# Pose Estimation and 3D Reconstruction

Following camera calibration, the system performs markerless human pose estimation using **MediaPipe Pose**, which detects anatomical landmarks from synchronized camera streams in real time.

Each camera independently estimates two-dimensional joint coordinates. These observations are then associated across multiple camera viewpoints to establish geometric correspondence for three-dimensional reconstruction.

## Pose Estimation

MediaPipe Pose was used to estimate key body landmarks including the head, shoulders, elbows, wrists, hips, knees, and ankles from RGB images captured simultaneously by multiple cameras.

The detected landmarks provide reliable two-dimensional image coordinates that serve as the basis for multi-view reconstruction.

## Multi-View Triangulation

Corresponding landmark observations from multiple calibrated cameras were combined using multi-view geometry to estimate three-dimensional joint positions.

The reconstruction process relies on calibrated camera parameters and projection matrices to recover spatial coordinates from multiple synchronized viewpoints.

## Kinematic Analysis

The reconstructed three-dimensional joint positions enable quantitative analysis of human movement, including:

- Joint trajectory visualization
- Limb motion tracking
- Spatial movement analysis
- Motion reconstruction across multiple viewpoints

The modular implementation allows future integration of additional computer vision models and more advanced reconstruction algorithms without modifying the overall system architecture.
---

# Repository Structure

```
Multi-Camera-Motion-Capturing-and-Kinematic-Analysis/
│
├── calibration/              # Camera calibration data
├── calibration_images/       # Checkerboard calibration images
├── data/                     # Input datasets
├── docs/                     # Research paper and presentation
├── images/                   # Figures and project images
├── models/                   # Model files
├── output/                   # Generated outputs
├── src/                      # Source code
├── stereo_images/            # Stereo image pairs
├── videos/                   # Demonstration videos
│
├── README.md
├── LICENSE
└── requirements.txt
```
---

# Installation

Clone the repository:

```bash
git clone https://github.com/kumarabhishek24-cpu/Multi-Camera-Motion-Capturing-and-Kinematic-Analysis.git

cd Multi-Camera-Motion-Capturing-and-Kinematic-Analysis
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```
