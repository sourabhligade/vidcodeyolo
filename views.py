from django.shortcuts import render
from .forms import VideoUploadForm, ImageUploadForm
from .models import UploadedVideo
import cv2
import numpy as np
from ultralytics import YOLO
import os
from django.conf import settings
import random
import pandas as pd

# Load color data
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colordetection/data/colors.csv', names=index, header=None)

# Load YOLO model
model = YOLO('yolov8n.pt')

def find_dominant_color(image):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5
    _, labels, centroids = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centroids = np.uint8(centroids)
    counts = np.bincount(labels.flatten())
    dominant_color_bgr = centroids[np.argmax(counts)]
    dominant_color_rgb = dominant_color_bgr[::-1]
    return dominant_color_rgb

def get_color_name(R, G, B):
    minimum = 10000
    cname = "Unknown"
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 0))
        colors.append(color)
    return colors

def yolo_object_detection(image_path):
    results = model(image_path)
    detection_results = []
    image = cv2.imread(image_path)
    detection_count = 1
    colors = generate_colors(len(results[0].boxes))

    for result in results:
        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            roi = image[y1:y2, x1:x2]
            dominant_color_rgb = find_dominant_color(roi)
            nearest_color_name = get_color_name(*dominant_color_rgb)

            detection_results.append({
                'number': detection_count,
                'label': label,
                'confidence': conf,
                'box': (x1, y1, x2 - x1, y2 - y1),
                'dominant_color': dominant_color_rgb,
                'color_name': nearest_color_name
            })

            color = colors[idx]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_background = (x1, y1 - label_size[1] - 10, x1 + label_size[0], y1 - 10)
            cv2.rectangle(image, (label_background[0], label_background[1]), 
                          (label_background[2], label_background[3]), color, cv2.FILLED)
            cv2.putText(image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, str(detection_count), (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            detection_count += 1
    
    result_image_path = os.path.join(settings.MEDIA_ROOT, 'results', os.path.basename(image_path).replace(".jpg", "_yolo.jpg"))
    os.makedirs(os.path.dirname(result_image_path), exist_ok=True)
    cv2.imwrite(result_image_path, image)
    result_image_url = settings.MEDIA_URL + 'results/' + os.path.basename(result_image_path)

    return result_image_url, detection_results

def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            video_path = form.instance.video.path
            frames_results = process_video(video_path)

            # Debug print statements
            print("Frames results:", frames_results)

            return render(request, 'colordetection/video_result.html', {
                'frames_results': frames_results
            })
        else:
            print("Form is invalid", form.errors)
    else:
        form = VideoUploadForm()
    
    print("Rendering upload video form")
    return render(request, 'colordetection/upload_video.html', {'form': form})




def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(settings.MEDIA_ROOT, f'temp_frame_{frame_count}.jpg')
        cv2.imwrite(frame_path, frame)
        result_image_url, detection_results = yolo_object_detection(frame_path)
        
        frames_results.append({
            'frame': frame_count,
            'detection_results': detection_results,
            'result_image_url': result_image_url
        })
        
        os.remove(frame_path)
        frame_count += 1

    cap.release()
    return frames_results

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            image_path = form.instance.image.path
            result_image_url, detection_results = yolo_object_detection(image_path)

            return render(request, 'colordetection/result.html', {
                'result_image_url': result_image_url,
                'detections': detection_results
            })
    else:
        form = ImageUploadForm()
    
    return render(request, 'colordetection/upload.html', {'form': form})
