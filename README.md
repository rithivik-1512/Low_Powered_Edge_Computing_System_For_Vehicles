# LoPECS: Computational Intelligence (CI) Module

This repository contains the **Computational Intelligence (CI) module** of the LoPECS project for autonomous vehicles. It focuses on perception and decision-making using AI algorithms for **real-time understanding of the environment**.

## Overview

The CI module in LoPECS implements key autonomous vehicle functionalities including:

* **Object Detection:** YOLO-based detection for recognizing vehicles, pedestrians, and obstacles.
* **Visual Odometry / SLAM:** Estimating vehicle motion and building maps using camera input.
* **Lane Detection:** Detecting road lanes for navigation.
* **Voice Recognition:** Using **Vosk** for processing voice commands.

This module demonstrates how **computational intelligence algorithms** can enable real-time perception and decision-making in autonomous vehicles.

## Features

* **YOLO Object Detection**: Real-time detection of dynamic and static objects on the road.
* **Visual Odometry & SLAM**: Tracks vehicle movement and constructs a 3D map from camera frames.
* **Lane Detection**: Detects lane boundaries using image processing techniques for navigation.
* **Voice Commands**: Recognizes predefined commands using Vosk speech recognition.

## Requirements

* Python 3.x
* Libraries:

```bash
numpy
opencv-python
torch
vosk
matplotlib
scikit-image
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone <repository_url>
cd LoPECS-CI
```

2. Run individual scripts for each module:

   * YOLO Object Detection: `python yolo_detection.py`
   * Visual Odometry / SLAM: `python visual_odometry.py`
   * Lane Detection: `python lane_detection.py`
   * Voice Recognition: `python voice_recognition.py`

3. Modify paths or video/camera inputs as needed.

## Folder Structure

```
LoPECS-CI/
│
├── yolo_detection.py
├── visual_odometry.py
├── lane_detection.py
├── voice_recognition.py
├── datasets/               # Optional sample videos or images
├── models/                 # Pre-trained YOLO or SLAM models
└── README.md
```

## Notes

* The focus of this repository is **perception and CI algorithms**, not scheduling or energy optimization.
* Can be integrated into a larger LoPECS framework for full autonomous vehicle simulations.

## Contributing

This repository is public and contributions are welcome! You can:

* Add new perception algorithms
* Improve accuracy of existing models
* Integrate additional datasets for testing

To contribute:

1. Fork the repo
2. Create a new branch (`git checkout -b feature-name`)
3. Make your changes
4. Commit and push (`git commit -m "Your message"`)
5. Open a Pull Request

---
