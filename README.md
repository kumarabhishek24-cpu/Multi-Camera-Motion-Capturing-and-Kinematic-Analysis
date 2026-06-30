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
