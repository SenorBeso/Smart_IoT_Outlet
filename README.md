# Smart_IoT_Outlet
## Overview

SmartHursey IoT Safety Outlet Node is a computer vision and current-sensing IoT system designed for safe control of a 120V outlet in a lab environment.

The system uses:

- A Windows workstation running YOLO object detection through `camera.py`
- MQTT wireless communication
- A Raspberry Pi Zero 2 W running `relay_pi_subscriber.py`
- An SCT current sensor connected through an ADS1115 ADC
- A GPIO-controlled relay connected to a 120V outlet

The goal of the system is to control outlet power based on whether a human is present. It also will shutdown the outlet if unsafe power draw exceeds certain thresholds.

---

## System Architecture

```

  [USB Camera]
       ↓
  [Windows Workstation]
  YOLO person detection using camera.py
       ↓ MQTT message with token
  [Raspberry Pi Zero 2 W]
  relay_pi_subscriber.py
       ↓
  [SCT Current Sensor + ADS1115]
       ↓
  [Edge Decision Logic]
       ↓
  [GPIO18 / Physical Pin 12]
       ↓
  [Relay-Controlled Outlet]
       ↓
  [SQLite Database]

```

