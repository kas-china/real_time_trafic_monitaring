import os
import logging
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
from threading import Lock
from datetime import datetime, timedelta
import sqlite3
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, make_response
from werkzeug.utils import secure_filename
import csv
from io import StringIO, BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import base64
import ffmpeg 
from flask import abort
app = Flask(__name__)
app.secret_key = '123'

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['CHARTS_FOLDER'] = 'static/charts'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
app.config['ALERT_WINDOW'] = 30  # Seconds for alert history
app.config['ALERT_THRESHOLD'] = 5.0  # Default vehicles per second

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logging.getLogger('PIL').setLevel(logging.INFO)

# Initialize global variables
camera = None
camera_index = 0
processing = False
last_detection_time = None
detection_interval = 2
detection_counts = defaultdict(int)
count_lock = Lock()
use_tracker = False
alert_history = []
model = YOLO('yolov8n.pt')
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
non_vehicle_classes = ['person', 'bicycle', 'dog', 'cat']

# Database initialization
def init_db():
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        c.execute("PRAGMA foreign_keys = ON")
        c.execute('''CREATE TABLE IF NOT EXISTS detections
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                     type TEXT,
                     car INTEGER DEFAULT 0,
                     truck INTEGER DEFAULT 0,
                     bus INTEGER DEFAULT 0,
                     motorcycle INTEGER DEFAULT 0,
                     total INTEGER DEFAULT 0,
                     file_path TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS alerts
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                     message TEXT,
                     alert_type TEXT)''')
        conn.commit()
        cleanup_db()
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        try:
            os.remove('detections.db')
            init_db()
        except Exception as ex:
            logging.error(f"Could not recreate database: {ex}")
            raise
    finally:
        if conn:
            conn.close()

def get_db_connection():
    conn = sqlite3.connect('detections.db')
    conn.row_factory = sqlite3.Row
    return conn

def cleanup_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        six_months_ago = datetime.now() - timedelta(days=180)
        c.execute("DELETE FROM detections WHERE timestamp < ?", (six_months_ago,))
        c.execute("DELETE FROM alerts WHERE timestamp < ?", (six_months_ago,))
        conn.commit()
        logging.info(f"Deleted records older than {six_months_ago}")
    except sqlite3.Error as e:
        logging.error(f"Database cleanup error: {e}")
    finally:
        if conn:
            conn.close()

init_db()

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHARTS_FOLDER'], exist_ok=True)

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def update_counts(class_name):
    with count_lock:
        detection_counts[class_name] += 1

def reset_counts():
    with count_lock:
        detection_counts.clear()

def save_detection_to_db(file_type, counts, file_path=None):
    total = sum(counts.values())
    conn = None
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''INSERT INTO detections 
                    (type, car, truck, bus, motorcycle, total, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                 (file_type, 
                  counts.get('car', 0),
                  counts.get('truck', 0),
                  counts.get('bus', 0),
                  counts.get('motorcycle', 0),
                  total,
                  file_path))
        conn.commit()
        logging.debug(f"Saved detection to DB: type={file_type}, counts={dict(counts)}, file_path={file_path}")
        return True
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_alert(message, alert_type='threshold'):
    conn = None
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('INSERT INTO alerts (message, alert_type) VALUES (?, ?)', (message, alert_type))
        conn.commit()
        logging.info(f"Alert saved: {message}, type: {alert_type}")
    except sqlite3.Error as e:
        logging.error(f"Alert save error: {e}")
    finally:
        if conn:
            conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alert_dashboard')
def alert_dashboard():
    return render_template('alert_dashboard.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    # Map extensions to MIME types
    mime_types = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png'
    }
    
    ext = os.path.splitext(filename)[1].lower()
    content_type = mime_types.get(ext, 'application/octet-stream')
    
    # Ensure the file exists
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(file_path):
       abort(404)
    
    response = send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    response.headers['Content-Type'] = content_type
    response.headers['Content-Disposition'] = f'inline; filename="{filename}"'
    
    # For MP4 files, ensure they can be streamed
    if ext == '.mp4':
        response.headers['Accept-Ranges'] = 'bytes'
    
    logging.debug(f"Serving processed file: {filename} with Content-Type: {content_type}")
    return response

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({'status': 'error', 'message': 'No file part in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        logging.error(f"File type not allowed: {file.filename}")
        return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400
    
    try:
        reset_counts()
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file and verify
        file.save(upload_path)
        if not os.path.exists(upload_path) or os.path.getsize(upload_path) == 0:
            logging.error(f"Failed to save uploaded file: {upload_path}")
            return jsonify({'status': 'error', 'message': 'Failed to save uploaded file'}), 500
        
        logging.debug(f"File saved successfully: {upload_path}")
        
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            counts = process_image(upload_path, processed_path)
            save_detection_to_db('image', counts, f"/processed/{processed_filename}")
            return jsonify({
                'status': 'success',
                'message': 'Image processed successfully',
                'original_url': f'/uploads/{filename}',
                'processed_url': f'/processed/{processed_filename}',
                'counts': dict(counts),
                'type': 'image',
                'frame_count': 1
            })
        else:
            result = process_video(upload_path, processed_path)
            processed_filename = os.path.basename(result['output_path'])  # Use the updated .mp4 filename
            save_detection_to_db('video', result['counts'], f"/processed/{processed_filename}")
            return jsonify({
                'status': 'success',
                'message': 'Video processed successfully',
                'original_url': f'/uploads/{filename}',
                'processed_url': f'/processed/{processed_filename}',
                'counts': dict(result['counts']),
                'type': 'video',
                'frame_count': result['frame_count']
            })
    except Exception as e:
        logging.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'Upload failed: {str(e)}'}), 500

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        logging.error(f"Could not read image: {input_path}")
        raise ValueError("Could not read image file")
    
    counts = defaultdict(int)
    tracked_ids = set()
    logging.debug(f"Processing image: {input_path}, Tracker: {'ON' if use_tracker else 'OFF'}")
    
    if use_tracker:
        results = model.track(img, persist=True, conf=0.5)
        logging.debug(f"Tracker enabled, results: {len(results)}")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                if class_name in vehicle_classes and box.id is not None:
                    track_id = int(box.id)
                    if track_id not in tracked_ids:
                        counts[class_name] += 1
                        tracked_ids.add(track_id)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"{class_name} ID:{track_id}"
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
                        cv2.putText(img, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        logging.debug(f"Tracked: {class_name}, ID: {track_id}, Box: ({x1}, {y1}, {x2}, {y2})")
                elif class_name in non_vehicle_classes:
                    save_alert(f"Unallowed track detected: {class_name}", "unallowed")
    else:
        results = model(img, conf=0.5)
        logging.debug(f"Tracker disabled, results: {len(results)}")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                if class_name in vehicle_classes:
                    counts[class_name] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = f"{class_name} {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
                    cv2.putText(img, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    logging.debug(f"Detected: {class_name}, conf: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")
                elif class_name in non_vehicle_classes:
                    save_alert(f"Unallowed detection: {class_name}", "unallowed")
    
    if not cv2.imwrite(output_path, img):
        logging.error(f"Failed to save image: {output_path}")
        raise ValueError("Failed to save processed image")
    
    logging.debug(f"Image saved: {output_path}, Counts: {dict(counts)}")
    return counts

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {input_path}")
        raise ValueError("Could not open video file")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Ensure output is always .mp4
    output_path = os.path.splitext(output_path)[0] + '.mp4'
    temp_output_path = os.path.splitext(output_path)[0] + '_temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v as a fallback
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        logging.error(f"Could not initialize video writer: {temp_output_path}")
        raise ValueError("Could not initialize video writer")
    
    counts = defaultdict(int)
    frame_count = 0
    tracked_ids = set()
    
    logging.debug(f"Processing video: {input_path}, Tracker: {'ON' if use_tracker else 'OFF'}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counts = defaultdict(int)
        if use_tracker:
            results = model.track(frame, persist=True, conf=0.5)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    if class_name in vehicle_classes and box.id is not None:
                        track_id = int(box.id)
                        if track_id not in tracked_ids:
                            frame_counts[class_name] += 1
                            tracked_ids.add(track_id)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = f"{class_name} ID:{track_id}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            logging.debug(f"Frame {frame_count} - Tracked: {class_name}, ID: {track_id}, Box: ({x1}, {y1}, {x2}, {y2})")
                    elif class_name in non_vehicle_classes:
                        save_alert(f"Unallowed track detected: {class_name}", "unallowed")
        else:
            results = model(frame, conf=0.5)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    if class_name in vehicle_classes:
                        frame_counts[class_name] += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = f"{class_name} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        logging.debug(f"Frame {frame_count} - Detected: {class_name}, conf: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")
                    elif class_name in non_vehicle_classes:
                        save_alert(f"Unallowed detection: {class_name}", "unallowed")
        
        for vehicle in vehicle_classes:
            counts[vehicle] = max(counts[vehicle], frame_counts[vehicle])
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
        logging.error(f"Failed to write temporary video: {temp_output_path}")
        raise ValueError("Failed to write temporary video")
    
    # Re-encode with ffmpeg-python to ensure H.264 compatibility
    try:
        (
            ffmpeg
            .input(temp_output_path)
            .output(
                output_path,
                vcodec='libx264',
                pix_fmt='yuv420p',  # Ensures compatibility with most browsers
                movflags='+faststart',  # Allows streaming
                acodec='aac',
                audio_bitrate='192k',
                crf=23,  
                preset='fast'  
            )
            .global_args('-hide_banner')  
            .global_args('-loglevel', 'error')  
            .run(overwrite_output=True)
        )
        logging.debug(f"Re-encoded video with FFmpeg: {output_path}")
    except ffmpeg.Error as e:
        logging.warning(f"FFmpeg re-encoding failed: {e.stderr.decode('utf8')}. Using OpenCV output: {temp_output_path}")
        output_path = temp_output_path  
    finally:
        if os.path.exists(temp_output_path) and temp_output_path != output_path:
            os.remove(temp_output_path)
    
    # Set proper permissions for the web server
    os.chmod(output_path, 0o644)
    
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        logging.error(f"Failed to write final video: {output_path}")
        raise ValueError("Failed to write final processed video")
    
    logging.debug(f"Video saved: {output_path}, Frame count: {frame_count}, Counts: {dict(counts)}")
    return {'counts': counts, 'frame_count': frame_count, 'output_path': output_path}

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    try:
        threshold = float(request.form.get('threshold', 0))
        if threshold < 0:
            return jsonify({'status': 'error', 'message': 'Threshold must be non-negative'}), 400
        app.config['ALERT_THRESHOLD'] = threshold
        logging.info(f"Alert threshold set to {threshold}")
        return jsonify({'status': 'success', 'message': f'Threshold set to {threshold}'})
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid threshold value'}), 400
    except Exception as e:
        logging.error(f"Set threshold error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_threshold')
def get_threshold():
    try:
        return jsonify({'status': 'success', 'threshold': app.config['ALERT_THRESHOLD']})
    except Exception as e:
        logging.error(f"Get threshold error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/export_csv')
def export_csv():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""SELECT 
                    strftime('%Y-%m-%d %H:%M:%S', timestamp) as date,
                    type, car, truck, bus, motorcycle, total
                    FROM detections ORDER BY timestamp DESC""")
        rows = c.fetchall()
        conn.close()
        
        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(['Timestamp', 'Type', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Total'])
        cw.writerows(rows)
        
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=detections_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    except Exception as e:
        logging.error(f"CSV export error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/export_pdf')
def export_pdf():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""SELECT 
                    strftime('%Y-%m-%d %H:%M:%S', timestamp) as date,
                    type, car, truck, bus, motorcycle, total
                    FROM detections ORDER BY timestamp DESC LIMIT 100""")
        rows = c.fetchall()
        conn.close()
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        styles = getSampleStyleSheet()
        elements.append(Paragraph("Vehicle Detection Report", styles['Title']))
        
        data = [['Timestamp', 'Type', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Total']]
        data.extend(rows)
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        
        doc.build(elements)
        buffer.seek(0)
        
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=detections_report.pdf'
        return response
    except Exception as e:
        logging.error(f"PDF export error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/radar_chart')
def radar_chart():
    try:
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT 
                strftime('%Y-%m-%d %H:%M', timestamp) as time_period,
                SUM(car) as car,
                SUM(truck) as truck,
                SUM(bus) as bus,
                SUM(motorcycle) as motorcycle
            FROM detections
            GROUP BY strftime('%Y-%m-%d %H:%M', timestamp)
            ORDER BY timestamp DESC
            LIMIT 5
        """, conn)
        conn.close()
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data available'}), 404
        
        categories = vehicle_classes
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        for _, row in df.iterrows():
            values = [row[c] for c in categories]
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=row['time_period'])
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        plt.title('Vehicle Detection Comparison', size=20, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify({'status': 'success', 'image': image_base64})
    except Exception as e:
        logging.error(f"Radar chart error: {e}")
        plt.close()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/line_chart')
def line_chart():
    try:
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT 
                strftime('%Y-%m-%d %H:%M:%S', timestamp) as timestamp,
                total
            FROM detections
            WHERE type IN ('real-time', 'image', 'video')
            ORDER BY timestamp
            LIMIT 100
        """, conn)
        conn.close()
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data available'}), 404
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['total'], marker='o')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        plt.title('Vehicle Detections Over Time')
        plt.xlabel('Time')
        plt.ylabel('Total Vehicles Detected')
        plt.grid(True)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify({'status': 'success', 'image': image_base64})
    except Exception as e:
        logging.error(f"Line chart error: {e}")
        plt.close()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/bar_chart')
def bar_chart():
    try:
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT 
                SUM(car) as car,
                SUM(truck) as truck,
                SUM(bus) as bus,
                SUM(motorcycle) as motorcycle
            FROM detections
            WHERE type IN ('real-time', 'image', 'video')
        """, conn)
        conn.close()
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data available'}), 404
        
        plt.figure(figsize=(8, 5))
        categories = vehicle_classes
        counts = [df[c][0] for c in categories]
        bars = plt.bar(categories, counts, color=['blue', 'green', 'red', 'purple'])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    '%d' % int(height), ha='center', va='bottom')
        
        plt.title('Total Vehicle Counts by Type')
        plt.xlabel('Vehicle Type')
        plt.ylabel('Count')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify({'status': 'success', 'image': image_base64})
    except Exception as e:
        logging.error(f"Bar chart error: {e}")
        plt.close()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/truck_chart')
def truck_chart():
    try:
        conn = get_db_connection()
        df = pd.read_sql("""
            SELECT 
                strftime('%Y-%m-%d %H:%M:%S', timestamp) as timestamp,
                truck
            FROM detections
            WHERE type IN ('real-time', 'image', 'video')
            ORDER BY timestamp
            LIMIT 100
        """, conn)
        conn.close()
        
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No truck data available'}), 404
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        plt.figure(figsize=(10, 5))
        plt.plot(df['timestamp'], df['truck'], marker='o', color='green')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        plt.title('Truck Detections Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Trucks Detected')
        plt.grid(True)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify({'status': 'success', 'image': image_base64})
    except Exception as e:
        logging.error(f"Truck chart error: {e}")
        plt.close()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download_chart/<chart_type>')
def download_chart(chart_type):
    try:
        if chart_type not in ['radar', 'line', 'bar', 'truck']:
            return jsonify({'status': 'error', 'message': 'Invalid chart type'}), 400
        
        conn = get_db_connection()
        if chart_type == 'radar':
            df = pd.read_sql("""
                SELECT 
                    strftime('%Y-%m-%d %H:%M', timestamp) as time_period,
                    SUM(car) as car,
                    SUM(truck) as truck,
                    SUM(bus) as bus,
                    SUM(motorcycle) as motorcycle
                FROM detections
                WHERE type IN ('real-time', 'image', 'video')
                GROUP BY strftime('%Y-%m-%d %H:%M', timestamp)
                ORDER BY timestamp DESC
                LIMIT 5
            """, conn)
            if df.empty:
                return jsonify({'status': 'error', 'message': 'No data available'}), 404
            categories = vehicle_classes
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)
            for _, row in df.iterrows():
                values = [row[c] for c in categories]
                values += values[:1]
                ax.plot(angles, values, linewidth=1, linestyle='solid', label=row['time_period'])
                ax.fill(angles, values, alpha=0.1)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            plt.title('Vehicle Detection Comparison', size=20, y=1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        elif chart_type == 'line':
            df = pd.read_sql("""
                SELECT 
                    strftime('%Y-%m-%d %H:%M:%S', timestamp) as timestamp,
                    total
                FROM detections
                WHERE type IN ('real-time', 'image', 'video')
                ORDER BY timestamp
                LIMIT 100
            """, conn)
            if df.empty:
                return jsonify({'status': 'error', 'message': 'No data available'}), 404
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            plt.figure(figsize=(10, 5))
            plt.plot(df['timestamp'], df['total'], marker='o')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            plt.title('Vehicle Detections Over Time')
            plt.xlabel('Time')
            plt.ylabel('Total Vehicles Detected')
            plt.grid(True)
        elif chart_type == 'bar':
            df = pd.read_sql("""
                SELECT 
                    SUM(car) as car,
                    SUM(truck) as truck,
                    SUM(bus) as bus,
                    SUM(motorcycle) as motorcycle
                FROM detections
                WHERE type IN ('real-time', 'image', 'video')
            """, conn)
            if df.empty:
                return jsonify({'status': 'error', 'message': 'No data available'}), 404
            plt.figure(figsize=(8, 5))
            categories = vehicle_classes
            counts = [df[c][0] for c in categories]
            bars = plt.bar(categories, counts, color=['blue', 'green', 'red', 'purple'])
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        '%d' % int(height), ha='center', va='bottom')
            plt.title('Total Vehicle Counts by Type')
            plt.xlabel('Vehicle Type')
            plt.ylabel('Count')
        elif chart_type == 'truck':
            df = pd.read_sql("""
                SELECT 
                    strftime('%Y-%m-%d %H:%M:%S', timestamp) as timestamp,
                    truck
                FROM detections
                WHERE type IN ('real-time', 'image', 'video')
                ORDER BY timestamp
                LIMIT 100
            """, conn)
            if df.empty:
                return jsonify({'status': 'error', 'message': 'No truck data available'}), 404
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            plt.figure(figsize=(10, 5))
            plt.plot(df['timestamp'], df['truck'], marker='o', color='green')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            plt.title('Truck Detections Over Time')
            plt.xlabel('Time')
            plt.ylabel('Number of Trucks Detected')
            plt.grid(True)
        
        conn.close()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'image/png'
        response.headers['Content-Disposition'] = f'attachment; filename={chart_type}_chart.png'
        return response
    except Exception as e:
        logging.error(f"Download chart error: {e}")
        plt.close()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_detection_details/<int:detection_id>')
def get_detection_details(detection_id):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            SELECT id, 
                strftime('%Y-%m-%d %H:%M:%S', timestamp) as date,
                type, car, truck, bus, motorcycle, total, file_path
            FROM detections WHERE id = ?
        """, (detection_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return jsonify({
                'status': 'success',
                'data': {
                    'id': row['id'],
                    'date': row['date'],
                    'type': row['type'],
                    'car': row['car'],
                    'truck': row['truck'],
                    'bus': row['bus'],
                    'motorcycle': row['motorcycle'],
                    'total': row['total'],
                    'file_path': row['file_path']
                }
            })
        return jsonify({'status': 'error', 'message': 'Detection not found'}), 404
    except Exception as e:
        logging.error(f"Detection details error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_camera', methods=['POST'])
def switch_camera():
    global camera, camera_index, processing
    try:
        camera_index = int(request.form.get('camera', 0))
        if camera_index not in [0, 1]:
            logging.error(f"Invalid camera index: {camera_index}")
            return jsonify({'status': 'error', 'message': 'Invalid camera index'}), 400
        
        if camera is not None and camera.isOpened():
            camera.release()
            logging.debug(f"Released camera index {camera_index}")
        
        camera = None
        if processing:
            camera = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
            if not camera.isOpened():
                processing = False
                logging.error(f"Could not open camera {camera_index}")
                return jsonify({'status': 'error', 'message': 'Could not open camera'}), 500
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logging.info(f"Switched to camera index {camera_index}")
        
        return jsonify({'status': 'success', 'message': f'Switched to camera {camera_index}'})
    except Exception as e:
        logging.error(f"Switch camera error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/start_camera')
def start_camera():
    global camera, processing, camera_index
    logging.debug(f"Starting camera at index {camera_index}")
    
    try:
        if camera is not None and camera.isOpened():
            camera.release()
        
        camera = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
        if not camera.isOpened():
            logging.error(f"Could not open camera index {camera_index}")
            return jsonify({'status': 'error', 'message': 'Could not open camera'}), 500
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        processing = True
        reset_counts()
        logging.info(f"Camera started at index {camera_index}")
        return jsonify({'status': 'success', 'message': 'Camera started'})
    except Exception as e:
        logging.error(f"Start camera error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop_camera')
def stop_camera():
    global camera, processing
    try:
        if camera is not None and camera.isOpened():
            camera.release()
            camera = None
            processing = False
            logging.info("Camera stopped")
        return jsonify({'status': 'success', 'message': 'Camera stopped'})
    except Exception as e:
        logging.error(f"Stop camera error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/toggle_tracking', methods=['POST'])
def toggle_tracking():
    global use_tracker
    try:
        use_tracker = request.form.get('enable', type=lambda x: x.lower() == 'true')
        reset_counts()
        logging.info(f"Tracking {'enabled' if use_tracker else 'disabled'}")
        return jsonify({
            'status': 'success',
            'tracking_enabled': use_tracker,
            'message': f"Tracking {'enabled' if use_tracker else 'disabled'}"
        })
    except Exception as e:
        logging.error(f"Toggle tracking error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_alerts')
def get_alerts():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            SELECT id, strftime('%Y-%m-%d %H:%M:%S', timestamp) as timestamp, 
                   message, alert_type
            FROM alerts ORDER BY timestamp DESC LIMIT 50
        """)
        alerts = [{
            'id': row['id'],
            'timestamp': row['timestamp'],
            'message': row['message'],
            'alert_type': row['alert_type']
        } for row in c.fetchall()]
        conn.close()
        return jsonify(alerts)
    except Exception as e:
        logging.error(f"Get alerts error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def generate_frames():
    global camera, processing, last_detection_time, camera_index, use_tracker, alert_history
    tracked_ids = set()
    # Store detection data for persistent drawing
    tracked_objects = {}  # {track_id or index: {'class_name': str, 'bbox': (x1, y1, x2, y2), 'label': str, 'conf': float}}
    
    while True:
        if not processing or camera is None or not camera.isOpened():
            blank_frame = np.zeros((480, 640, 3), np.uint8)
            message = "Click 'Start Camera'" if not processing else "Camera unavailable"
            cv2.putText(blank_frame, message, (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.2)
            continue
        
        success, frame = camera.read()
        if not success:
            logging.error(f"Failed to read frame from camera {camera_index}")
            try:
                camera.release()
                camera = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
                if not camera.isOpened():
                    logging.error(f"Camera {camera_index} failed to reinitialize")
                    camera = None
                    processing = False
                    continue
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                success, frame = camera.read()
                if not success:
                    logging.error("Still failed to read frame")
                    continue
            except Exception as e:
                logging.error(f"Camera reinitialization error: {e}")
                camera = None
                processing = False
                continue
        
        cv2.putText(frame, f"Tracking: {'ON' if use_tracker else 'OFF'}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if use_tracker else (0, 0, 255), 2)
        cv2.putText(frame, f"Threshold: {app.config['ALERT_THRESHOLD']}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Removed detection_interval check for live detection on every frame
        try:
            current_counts = defaultdict(int)
            results = model.track(frame, persist=True, conf=0.5) if use_tracker else model(frame, conf=0.5)
            
            # Clear previous objects
            tracked_objects.clear()
            
            for result in results:
                for idx, box in enumerate(result.boxes):
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if class_name in vehicle_classes:
                        if use_tracker and box.id is not None:
                            track_id = int(box.id)
                            if track_id not in tracked_ids:
                                current_counts[class_name] += 1
                                update_counts(class_name)
                                tracked_ids.add(track_id)
                                label = f"{class_name} ID:{track_id}"
                                # Store object for drawing
                                tracked_objects[track_id] = {
                                    'class_name': class_name,
                                    'bbox': (x1, y1, x2, y2),
                                    'label': label,
                                    'conf': conf
                                }
                                logging.debug(f"Tracked: {class_name}, ID: {track_id}, Box: ({x1}, {y1}, {x2}, {y2})")
                        elif not use_tracker:
                            current_counts[class_name] += 1
                            update_counts(class_name)
                            label = f"{class_name} {conf:.2f}"
                            # Store object for drawing (use index as key)
                            tracked_objects[idx] = {
                                'class_name': class_name,
                                'bbox': (x1, y1, x2, y2),
                                'label': f"{class_name} {conf:.2f}",
                                'conf': conf
                            }
                            logging.debug(f"Detected: {class_name}, conf: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")
                    
                    elif class_name in non_vehicle_classes:
                        save_alert(f"Unallowed class detected: {class_name}", "unallowed")
                        label = f"{class_name} {conf:.2f}"
                        # Store non-vehicle object
                        tracked_objects[f"non_vehicle_{idx}"] = {
                            'class_name': class_name,
                            'bbox': (x1, y1, x2, y2),
                            'label': label,
                            'conf': conf
                        }
            
            total = sum(current_counts.values())
            if total > 0:
                save_detection_to_db('real-time', current_counts)
            
            alert_history.append({'timestamp': time.time(), 'total': total})
            alert_history[:] = [h for h in alert_history if time.time() - h['timestamp'] <= app.config['ALERT_WINDOW']]
            
            if len(alert_history) > 1:
                time_span = alert_history[-1]['timestamp'] - alert_history[0]['timestamp']
                if time_span > 0:
                    rate = sum(h['total'] for h in alert_history) / time_span
                    if rate >= app.config['ALERT_THRESHOLD']:
                        save_alert(f"Threshold exceeded: {rate:.1f} vehicles/sec", "threshold")
            
            last_detection_time = time.time()
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
        
        # Draw boxes and labels for all stored objects on every frame
        for obj in tracked_objects.values():
            x1, y1, x2, y2 = obj['bbox']
            label = obj['label']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
@app.route('/get_counts')
def get_counts():
    try:
        with count_lock:
            counts = dict(detection_counts)
            total = sum(counts.values())
        return jsonify({'status': 'success', 'counts': counts, 'total': total})
    except Exception as e:
        logging.error(f"Get counts error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/reset_counts')
def reset_counts_endpoint():
    try:
        reset_counts()
        return jsonify({'status': 'success', 'message': 'Counts reset'})
    except Exception as e:
        logging.error(f"Reset counts error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_history')
def get_history():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            SELECT id, 
                strftime('%Y-%m-%d %H:%M:%S', timestamp) as date,
                type, car, truck, bus, motorcycle, total, file_path
            FROM detections ORDER BY timestamp DESC LIMIT 100
        """)
        rows = c.fetchall()
        conn.close()
        
        history = [{
            'id': row['id'],
            'date': row['date'],
            'type': row['type'],
            'car': row['car'],
            'truck': row['truck'],
            'bus': row['bus'],
            'motorcycle': row['motorcycle'],
            'total': row['total'],
            'file_path': row['file_path']
        } for row in rows]
        
        return jsonify(history)
    except Exception as e:
        logging.error(f"Get history error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_recent_detections')
def get_recent_detections():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            SELECT strftime('%Y-%m-%d %H:%M:%S', timestamp) as date,
                type, car, truck, bus, motorcycle
            FROM detections ORDER BY timestamp DESC LIMIT 50
        """)
        rows = c.fetchall()
        conn.close()
        
        detections = [{
            'timestamp': row['date'],
            'type': row['type'],
            'car': row['car'],
            'truck': row['truck'],
            'bus': row['bus'],
            'motorcycle': row['motorcycle']
        } for row in rows]
        
        return jsonify(detections)
    except Exception as e:
        logging.error(f"Get recent detections error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    max_retries = 3
    for attempt in range(max_retries):
        try:
            init_db()
            break
        except Exception as e:
            logging.error(f"Initialization failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logging.critical("Max retries reached. Exiting.")
                raise
            time.sleep(1)
    app.run(debug=True, host='0.0.0.0', port=5000)