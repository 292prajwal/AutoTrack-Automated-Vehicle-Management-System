from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import re
from datetime import datetime, timedelta
import easyocr
import threading
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///license_plates.db'
db = SQLAlchemy(app)

# Directory for uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# Load the YOLO model
model_path = 'best.pt'  # Update with your model path
model = YOLO(model_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Database models
class Plate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plate = db.Column(db.String(20), nullable=False)
    entry_time = db.Column(db.DateTime, nullable=False)
    exit_time = db.Column(db.DateTime)

    def __repr__(self):
        return f"<Plate {self.plate}>"

# Initialize SocketIO
socketio = SocketIO(app)

# Helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(img):
    results = model(img)
    detected_plates = []

    for result in results:
        boxes = result.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Draw a bounding box around the detected region
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box for license plate detection

            # Crop the box and apply OCR
            img_pil = Image.fromarray(img)
            cropped_image = img_pil.crop((x1, y1, x2, y2))
            image_np = np.array(cropped_image)

            # Perform OCR
            ocr_result = reader.readtext(image_np)
            result_text = ""
            max_confidence = 0.0
            for detection in ocr_result:
                text = detection[1]
                confidence = detection[2]
                result_text += text  # Concatenate texts
                if confidence > max_confidence:
                    max_confidence = confidence

            # Remove spaces and convert to uppercase
            result_text = result_text.replace(" ", "").upper().replace("-", "")

            # Search for license plate format
            match = re.search(r'\d{2}[A-Z]{1,3}\d{2,4}', result_text)

            if match and max_confidence > 0.75:
                plate_number = match.group(0)
                current_time = datetime.now()

                # Draw the license plate number and accuracy on the image
                accuracy_text = f"{plate_number} ({int(max_confidence * 100)}%)"
                cv2.putText(img, accuracy_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # Check if plate exists and handle entry/exit logic
                with app.app_context():
                    plate_record = Plate.query.filter_by(plate=plate_number).order_by(Plate.entry_time.desc()).first()

                    if plate_record is None or (plate_record.exit_time and current_time - plate_record.exit_time > timedelta(minutes=2)):
                        # New entry
                        new_plate = Plate(plate=plate_number, entry_time=current_time)
                        db.session.add(new_plate)
                        db.session.commit()
                        detected_plates.append({'plate': plate_number, 'entry_time': current_time, 'status': 'Entered'})
                    elif not plate_record.exit_time and current_time - plate_record.entry_time > timedelta(minutes=2):
                        # Mark exit
                        plate_record.exit_time = current_time
                        db.session.commit()
                        detected_plates.append({'plate': plate_number, 'exit_time': current_time, 'status': 'Exited'})

    return img, detected_plates

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Process the image
            img = cv2.imread(image_path)
            processed_img, detected_plates = process_image(img)
            result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_image_path, processed_img)

            # Return the processed image URL and detection details
            if detected_plates:
                plate_number = detected_plates[0]['plate']
                detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return jsonify({
                    'processed_image_url': result_image_path,
                    'plate_number': plate_number,
                    'detection_time': detection_time
                })
            else:
                return jsonify({'error': 'No plate detected'}), 400

    return render_template('upload_image.html')


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # Process the video in a separate thread to prevent blocking
            threading.Thread(target=process_video, args=(video_path,)).start()

            flash('Video processing started. Check back later.')
            return redirect(url_for('vehicles'))
    return render_template('upload_video.html')

@app.route('/live_cam')
def live_cam():
    return render_template('live_cam.html')

def gen_frames():
    # Use your camera source here (e.g., 0 for default webcam)
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            with app.app_context():
                processed_frame, _ = process_image(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def process_video(video_path):
    with app.app_context():
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, _ = process_image(frame)
            # Optionally save processed frames or output video
        cap.release()

@app.route('/vehicles')
def vehicles():
    plates = Plate.query.all()
    return render_template('vehicles.html', plates=plates)

@app.route('/mark_exit/<int:plate_id>')
def mark_exit(plate_id):
    plate_record = Plate.query.get_or_404(plate_id)
    if plate_record.exit_time is None:
        plate_record.exit_time = datetime.now()
        db.session.commit()
        flash('Exit time recorded.')
    else:
        flash('Exit time already recorded.')
    return redirect(url_for('vehicles'))

# For SocketIO handling
@socketio.on('connect')
def handle_connect():
    emit('message', {'data': 'Connected to the server'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)
