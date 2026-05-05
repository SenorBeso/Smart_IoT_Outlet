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

## Venv Requirements
With pip, you need to install a few libraries
Windows:
```
pip install --upgrade pip
pip install ultralytics
pip install opencv-python
pip install paho-mqtt
```

Raspberry Pi:
```
pip install --upgrade pip
pip install paho-mqtt
pip install gpiozero
pip install adafruit-blinka
pip install adafruit-circuitpython-ads1x15
pip install lgpio

```

## Starting the System

The system uses two scripts:

- `relay_pi_subscriber.py` runs on the Raspberry Pi Zero 2 W.
- `camera.py` runs on the Windows workstation.

Start the Raspberry Pi script first, then start the Windows camera script.

---

### 1. Start the Raspberry Pi Relay/Power Monitor

SSH into the Raspberry Pi or open a terminal on the Pi:

```bash
cd ~/Documents/iot
source venv/bin/activate
export MQTT_SHARED_TOKEN="password123"
python3 relay_pi_subscriber.py --relay-pin 18 --mqtt-token "$MQTT_SHARED_TOKEN"
```

To get the token or set it, you do this:
```
export MQTT_SHARED_TOKEN="password123"
```

### 2. Start the windows Camera script - replace with correct ip and correct token of the Pi - camera "0" is default, but you can set for other cameras as well.
```
python camera.py --camera 0 --mqtt-broker 192.168.4.204 --mqtt-token password123
```
