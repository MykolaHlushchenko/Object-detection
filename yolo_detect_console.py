import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float) # Changed to float for correct numerical comparison
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Setup bounding box colors (only needed if recording video)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
frame_idx = 0

print("=== STARTING PROCESSING (Press Ctrl+C in console to quit) ===")

# Begin inference loop
try:
    while True:
        t_start = time.perf_counter()
        frame_idx += 1

        # Load frame from image source
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('\nAll images have been processed. Exiting program.')
                sys.exit(0)
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count = img_count + 1
            source_info = f"File: {os.path.basename(img_filename)}"
        
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('\nReached end of the video file. Exiting program.')
                break
            source_info = f"Video Frame #{frame_idx}"
        
        elif source_type == 'usb':
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('\nUnable to read frames from the USB camera. Exiting program.')
                break
            source_info = f"USB Cam Frame #{frame_idx}"

        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if (frame is None):
                print('\nUnable to read frames from the Picamera. Exiting program.')
                break
            source_info = f"Picamera Frame #{frame_idx}"

        # Resize frame if requested
        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # Run inference on frame
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Gather information about detected objects for the console
        found_objects = []
        object_count = 0

        # Go through each detection
        for i in range(len(detections)):
            conf = detections[i].conf.item()

            # Filter by confidence threshold
            if conf >= min_thresh:
                classidx = int(detections[i].cls.item())
                classname = labels[classidx]
                object_count += 1
                found_objects.append(f"{classname} ({conf*100:.1f}%)")

                # Draw bounding boxes ONLY if recording is enabled
                if record:
                    xyxy_tensor = detections[i].xyxy.cpu()
                    xyxy = xyxy_tensor.numpy().squeeze()
                    xmin, ymin, xmax, ymax = xyxy.astype(int)
                    color = bbox_colors[classidx % 10]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    label = f'{classname}: {int(conf*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Calculate FPS for the current frame
        t_stop = time.perf_counter()
        frame_rate_calc = float(1 / (t_stop - t_start))

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = np.mean(frame_rate_buffer)

        # --- Console Output ---
        # Format the detected objects string
        objects_str = ", ".join(found_objects) if object_count > 0 else "Nothing detected"
        
        # Use \r for dynamic single-line updates during streaming (video/camera)
        if source_type in ['video', 'usb', 'picamera']:
            sys.stdout.write(f"\r[{source_info}] FPS: {avg_frame_rate:.1f} | Objects found: {object_count} [{objects_str}]" + " " * 20)
            sys.stdout.flush()
        else:
            # For static images, print each file on a new line
            print(f"[{source_info}] Objects found: {object_count} [{objects_str}]")

        # Save frame if recording is enabled
        if record:
            recorder.write(frame)

except KeyboardInterrupt:
    print('\nProgram stopped by user via Ctrl+C.')

# Clean up
print(f'\nAverage pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: 
    recorder.release()
    print("Video recording saved as 'demo1.avi'")
cv2.destroyAllWindows()
