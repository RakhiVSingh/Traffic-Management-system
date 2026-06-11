# 🚦 Smart Traffic Management System

An intelligent traffic control system that uses computer vision to detect vehicles and pedestrians, dynamically adjust traffic light timings, and record traffic footage for analysis.

## ✨ Features

- **Real-time Vehicle Detection** - Detects cars using Haar cascades
- **Pedestrian Detection** - Identifies pedestrians using HOG descriptors
- **Adaptive Traffic Lights** - Dynamically adjusts signal timing based on traffic volume
- **Emergency Vehicle Priority** - Automatically prioritizes emergency vehicles
- **Environmental Adaptations** - Adjusts timings for rain and low-light conditions
- **Video Recording** - Automatically records and saves traffic footage
- **Screenshot Capture** - Manual screenshot functionality
- **Interactive Display** - Real-time visualization with traffic light status

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `R` | Start/Stop manual recording |
| `S` | Take screenshot |

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python numpy
