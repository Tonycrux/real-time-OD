import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, Response

# Initialize Flask app
app = Flask(__name__)

# Threshold to detect object
thres = 0.56

# Labels for objects from COCO dataset
classNames = []
classFile = r'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Paths to model config and weights
configPath = r'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = r'frozen_inference_graph.pb'

# Defining the pre-trained model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Real-time camera feed
def generate_frames():
    cap = cv2.VideoCapture(0)  # Use camera (change to video file path if needed)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        success, img = cap.read()  # Capture frame-by-frame
        if not success:
            break

        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                # Draw bounding box and labels
                cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 55),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

        # Convert the image to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

# Flask route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Home route
@app.route('/')
def index():
    return '''
        <html>
            <body>
                <h1>Real-Time Object Detection</h1>
                <img src="/video_feed" width="1280" height="720"/>
            </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)