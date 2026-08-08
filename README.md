# MechaMinds

Contactless Neonatal Intensive Care Unit (NICU) Monitor

A non-contact, multi-modal vital sign monitoring system designed for fragile infants in Neonatal Intensive Care Units (NICUs). By fusing optical computer vision methods such as rPPG and EVM with millimeter-wave radar sensing, this system provides a research-oriented framework for real-time, non-invasive tracking of heart rate, oxygen saturation, and respiratory rate without physical contact, adhesive leads, or skin trauma.

## 🏗️ System Architecture

```text
                       ┌─────────────────────────────────────────┐
                       │  Contactless NICU Monitoring System     │
                       └────────────────────┬────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               │                                                         │
    ┌──────────▼──────────┐                                   ┌──────────▼──────────┐
    │   Raw Camera Feed   │                                   │   60GHz FMCW Radar  │
    └──────────┬──────────┘                                   └──────────┬──────────┘
               │                                                         │
   ┌───────────┴───────────┐                                 ┌───────────┴───────────┐
   │                       │                                 │                       │
┌──▼──┐                 ┌──▼──┐                           ┌──▼──┐                 ┌──▼──┐
│ EVM │                 │rPPG │                           │Chest│                 │  Artifact   │
└─┬───┘                 └─┬───┘                           │Displ│                 │    Check    │
  │ Visual Motion/        │ Data Extraction               └─┬───┘                 └─┬───┘
  │ Color Amp             │ & Signal Processing             │ Micro-                │ Body
  │                       │                                 │ Movements             │ Movements
  ▼                       ▼                                 ▼                       ▼
Continuous HR /        Continuous HR /                     Radar-based Chest
Color Amplification     SpO2 Trends                         Displacement Vector
   │                       │                                 │                       │
   └───────────────────────┴────────────────┬────────────────┴───────────────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │  FUSION & CROSS-       │
                               │  VALIDATION ENGINE      │
                               └────────────┬────────────┘
                                            │
                               ┌────────────▼────────────┘
                               │   NURSE DASHBOARD UI    │
                               └─────────────────────────┘
```

## 💡 Key Features and Modules

### 1. Optical Vision Pipeline

- rPPG (Remote Photoplethysmography): Extracts subtle color variations from facial skin regions to estimate continuous heart rate and oxygen saturation trends.
- EVM (Eulerian Video Magnification): Amplifies imperceptible color changes and small motion patterns for visual inspection of perfusion and breathing-related dynamics.

### 2. 60GHz FMCW Radar Pipeline

- Chest displacement tracking: Captures millimeter-scale chest wall motion from Doppler phase shifts to estimate respiratory rate.
- Artifact checking: Identifies gross body movement and environmental disturbances to improve signal quality and reduce false positives.

### 3. Fusion and Cross-Validation Engine

- Signal validation: Cross-checks optical signals with radar-based motion features to improve robustness.
- Alarm triggering: Supports real-time monitoring for apnea, bradycardia, and other distress events.

### 4. Nurse Dashboard UI

- Live EVM perfusion view
- Continuous vital sign readouts for HR, SpO2, and RR
- Real-time waveform visualization for pulse and respiratory signals

## 🛠️ Tech Stack and Requirements

- Language: Python 3.9+
- Computer vision and signal processing: OpenCV, NumPy, SciPy
- Web interface: Flask, HTML5, JavaScript
- Radar interface: Serial or UDP integration for 60GHz FMCW radar hardware

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MechaMinds.git
cd MechaMinds
```

### 2. Install dependencies

```bash
pip install flask opencv-python numpy scipy
```

### 3. Start the EVM service

```bash
python EVM.py
```

### 4. Start the rPPG service

```bash
python rPPG.py
```

### 5. Open the dashboard

Open the dashboard in a browser and allow camera access when prompted.

## 📁 Repository Structure

- EVM.py: Flask application for EVM-style processing and image exchange.
- rPPG.py: Flask application for rPPG-style processing and image exchange.
- dashboard.html: Browser-based dashboard for webcam capture and visualization.
- tests/: Regression tests for core processing pipeline behavior.

## 🧪 Testing

Run the test suite with:

```bash
pytest -q
```

## ⚠️ Disclaimer

This project is a research and development prototype. It is not intended for clinical diagnosis, treatment, or any use that replaces validated medical devices or regulated clinical workflows.