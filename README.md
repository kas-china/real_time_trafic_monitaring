# Real-Time Traffic Monitoring System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.0%2B-green.svg)

## Overview

This open-source **Real-Time Traffic Monitoring System** is a Flask-based web application designed for vehicle detection and tracking using the YOLOv8 model, OpenCV, and FFmpeg. It processes images, videos, and live camera feeds to detect vehicles (cars, trucks, buses, motorcycles), stores results in a SQLite database, and provides visualizations and reports for traffic analysis. The system is scalable, user-friendly, and adaptable for traffic management, surveillance, urban planning, and research.

**Authors:**
- Zeleke Kassahun Getachewu
- Degife Ruth Dachew 

**Submission Date:** May 28, 2025

## Table of Contents
1. [Introduction](#introduction)
2. [Purpose and Scope](#purpose-and-scope)
3. [Key Features](#key-features)
4. [Use Cases](#use-cases)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Technical Architecture](#technical-architecture)
8. [Performance Considerations](#performance-considerations)
9. [Limitations](#limitations)
10. [Future Improvements](#future-improvements)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction

This project provides a robust solution for real-time and offline vehicle detection and tracking. Built with Flask, OpenCV, and YOLOv8, it processes various input sources (images, videos, live feeds) and generates actionable insights through visualizations (radar, line, bar, truck charts) and reports (CSV, PDF). The system stores detection data in a SQLite database and includes an alert system for high vehicle activity, making it a versatile tool for traffic monitoring.

The open-source nature encourages community contributions, while its modular design supports scalability and adaptability across diverse environments.

## Purpose and Scope

The application aims to monitor vehicle types in real-time and offline scenarios, serving applications such as:
- Traffic management
- Urban planning
- Surveillance
- Research

**Scope:**
- Detect and track vehicles in images, videos, and live feeds with a confidence threshold of 0.5.
- Store detection results in a SQLite database for analysis.
- Generate CSV and PDF reports for data sharing.
- Visualize trends via charts to support decision-making.
- Trigger real-time alerts for high vehicle activity (> 5 vehicles/sec in a 30-second window).

The system supports a maximum file size of 100MB, adjustable based on hardware capabilities.

## Key Features

- **Vehicle Detection and Tracking**: Uses YOLOv8 (yolov8n.pt) for accurate detection with optional tracking, annotating outputs with red bounding boxes and labels (class name, confidence, track ID).
- **File Processing**: Handles image (PNG, JPG, JPEG) and video (MP4, AVI, MOV) uploads, saving annotated outputs in `static/processed`.
- **Database Storage**: Stores detections and alerts in `detections.db` (SQLite) for persistent querying.
- **Reporting**: Exports data as CSV (`/export_csv`) or PDF (`/export_pdf`) using ReportLab.
- **Visualization**: Generates four chart types:
  - **Radar Chart** (`/radar_chart`): Compares vehicle counts across 5 time periods.
  - **Line Chart** (`/line_chart`): Plots total detections over time (100 records).
  - **Bar Chart** (`/bar_chart`): Shows counts by vehicle type.
  - **Truck Chart** (`/truck_chart`): Tracks truck detections over time.
- **Real-Time Camera Feed**: Streams live feeds (`/video_feed`) with detection, supporting camera switching (indices 0, 1).
- **Alert System**: Triggers alerts for high activity, stored in the database and accessible via `/get_alerts`.
- **Logging**: DEBUG-level logs to console and `app.log` for debugging.
- **FFmpeg Integration**: Re-encodes videos to H.264 (libx264, yuv420p, faststart) for browser compatibility.
- **Thread Safety**: Uses `threading.Lock` for accurate counting in concurrent environments.

## Use Cases

- **Traffic Management**: Analyze congestion and optimize traffic signals.
- **Surveillance**: Monitor vehicles in parking lots, toll booths, or restricted areas.
- **Urban Planning**: Collect vehicle data for infrastructure planning.
- **Research**: Support studies in computer vision and traffic patterns.

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg installed and accessible in system PATH
- Git (optional, for cloning the repository)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/traffic-monitoring-system.git
   cd traffic-monitoring-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install flask opencv-python ultralytics numpy pandas matplotlib reportlab ffmpeg-python
   ```

3. **Download YOLOv8 Model**:
   - Download `yolov8n.pt` from the [Ultralytics YOLOv8 releases](https://github.com/ultralytics/ultralytics).
   - Place it in the project root directory.

4. **Create Directories**:
   ```bash
   mkdir -p static/uploads static/processed static/charts
   ```

5. **Install FFmpeg**:
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH.
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or equivalent.

6. **Initialize Database**:
   - The database (`detections.db`) is automatically created on first run.

7. **Run the Application**:
   ```bash
   python app.py
   ```
   - Access the web interface at `http://localhost:5000`.

## Usage

1. **Web Interface**:
   - Navigate to `http://localhost:5000`.
   - Upload images or videos via the `/upload` endpoint.
   - Control camera feed using `/start_camera`, `/stop_camera`, `/switch_camera`, and `/toggle_tracking`.

2. **Data Access**:
   - View detection counts: `/get_counts`
   - Retrieve history: `/get_history` (100 records)
   - Get recent detections: `/get_recent_detections` (50 records)
   - Fetch alerts: `/get_alerts` (50 records)
   - Access specific detection: `/get_detection_details/<id>`

3. **Reports and Visualizations**:
   - Export CSV: `/export_csv`
   - Export PDF: `/export_pdf`
   - View charts: `/radar_chart`, `/line_chart`, `/bar_chart`, `/truck_chart`
   - Download charts: `/download_chart/<chart_type>` (e.g., `truck`)

4. **Camera Feed**:
   - Start feed: `/start_camera`
   - View feed: `/video_feed`
   - Switch cameras: `/switch_camera` (POST with `camera=0` or `1`)
   - Toggle tracking: `/toggle_tracking` (POST with `enable=true/false`)

## Technical Architecture

### Dependencies
- Flask: Web framework
- OpenCV: Image/video processing
- Ultralytics YOLO: Vehicle detection/tracking
- FFmpeg: Video re-encoding
- NumPy: Array operations
- SQLite3: Database storage
- Pandas: Data manipulation
- Matplotlib: Chart generation
- ReportLab: PDF reports
- Werkzeug: Secure file handling

### Configuration
- **Folders**: `static/uploads`, `static/processed`, `static/charts`
- **File Limits**: 100MB max upload size
- **Allowed Extensions**: PNG, JPG, JPEG, MP4, AVI, MOV
- **Alert Settings**: 30-second window, 5 vehicles/sec threshold
- **YOLO Model**: yolov8n.pt, 0.5 confidence
- **Camera**: 640x480 resolution, 2-second detection interval
- **FFmpeg**: H.264, yuv420p, faststart

### Database Structure
- **detections**:
  - Columns: `id` (PK), `timestamp`, `type`, `car`, `truck`, `bus`, `motorcycle`, `total`, `file_path`
- **alerts**:
  - Columns: `id` (PK), `timestamp`, `message`, `alert_type`
- Cleanup: Removes records older than 180 days.

### File Processing
- **Images**: Processed with OpenCV, annotated with red bounding boxes, saved with `processed_` prefix.
- **Videos**: Frame-by-frame detection, re-encoded to H.264 via FFmpeg.

### Web Endpoints
See [report](#4.5-web-endpoints) for full list.

### Visualization
Charts are base64-encoded PNGs:
- **Radar**: Compares vehicle counts over 5 periods.
- **Line**: Total detections over time.
- **Bar**: Counts by vehicle type.

### Real-Time Camera Feed
- Streams via `/video_feed` with 2-second detection intervals.
- Supports camera switching and thread-safe counting.

### Alert System
- Triggers for > 5 vehicles/sec in 30 seconds.
- triger for humans and object that are not allowed on the road 
- Stored in `alerts` table, retrievable via `/get_alerts`.

### Logging
- DEBUG-level logs to console and `app.log`.

### Thread Safety
- Uses `threading.Lock` for concurrent count updates.

## Performance Considerations
- **Processing**: YOLOv8 and FFmpeg may be slow on low-end hardware.
- **Database**: SQLite optimized for small datasets (100-record limit).
- **Camera**: 640x480 resolution reduces overhead.
- **Scalability**: Single-server suitable; high traffic may require async processing.

## Limitations
- Slow performance on low-end hardware.
- Limited database cleanup.
- Supports only two camera indices.
- Security risks from hardcoded secret key and no authentication.


## Future Improvements
- Enable GPU support for YOLOv8.
- Enhance database cleanup.
- Support dynamic camera discovery.
- Add authentication and random secret key.
- Implement file cleanup.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Report issues or suggest features via the [Issues](https://github.com/kaschina/real_time_trafic_monitaring.git) tab.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Prepared by:**
- Zeleke Kassahun Getachewu
- Degife Ruth Dachew

**Contact**: [2120246060@mail.nankai.edu.cn] or [Repository Link](https://github.com/kaschina/real_time_trafic_monitaring.git)
