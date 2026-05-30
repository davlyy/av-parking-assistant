# 🚗 AV Parking Assistant

A real-time parking assistance system using computer vision and AI to guide a vehicle
into a parking spot. The system processes a live camera feed — either mounted on the
car itself or providing a bird's eye view — and predicts driving directions to assist
with precise parking maneuvers.

## Features

- Live video input via webcam, phone camera, or CARLA simulator
- AI-powered direction prediction from camera feed
- Parking spot boundary detection
- Vehicle position & trajectory estimation
- Real-time overlay with distance indicators and alignment cues
- Obstacle detection and proximity warnings
- Support for both physical RC-car and fully simulated environments

## Tech Stack

| Component | Technology |

| Computer Vision | OpenCV |
| AI / Object Detection | YOLOv8 |
| Rendering / UI Overlay | OGRE3D + ImGui |
| Simulation | CARLA |
| Camera | On-car cam or bird's eye view cam |
| Language | Python |

## Project Structure

parking-assistant/
├── src/
│   ├── detection/       
│   ├── overlay/                 
│   └── main.py
├── data/                # Datasets (gitignored)               
├── tests/
├── requirements.txt
└── README.md