###################### Contributors #############################
# Licensed under the Apache License, Version 2.0;
# Y.M.W.H.M.R.P.J.R.B.KIRIDANA, 248355U
# T.M.SEEDIN , 248375F
# K.S.RANGANA, 248367H
###################### Importing Libraries ######################
print("Loading Libraries...")
import time
start_time = time.time()
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import random
import torchvision.models as models
import torchvision.transforms as transforms
###################### Defining Paths ################################
print("Defining paths...")
img_output_dir = '/Users/punsisikiridana/Documents/University_Education/MSc/SEMESTER_2/IN24_S2_CS5801_Advanced_AI/Project/GRAPH/Graph-Graph/data_preparation/output_frames_obj_features'
npz_output_dir = '/Users/punsisikiridana/Documents/University_Education/MSc/SEMESTER_2/IN24_S2_CS5801_Advanced_AI/Project/GRAPH/Graph-Graph/data_preparation/npz_output'
input_folder_negative = '/Users/punsisikiridana/Documents/University_Education/MSc/SEMESTER_2/IN24_S2_CS5801_Advanced_AI/Project/GRAPH/Graph-Graph/data_preparation/input_clips/negative'
input_folder_positive = '/Users/punsisikiridana/Documents/University_Education/MSc/SEMESTER_2/IN24_S2_CS5801_Advanced_AI/Project/GRAPH/Graph-Graph/data_preparation/CCD2'
output_folder_dim = '/Users/punsisikiridana/Documents/University_Education/MSc/SEMESTER_2/IN24_S2_CS5801_Advanced_AI/Project/GRAPH/Graph-Graph/data_preparation/video_dimensions'
output_folder_global = '/Users/punsisikiridana/Documents/University_Education/MSc/SEMESTER_2/IN24_S2_CS5801_Advanced_AI/Project/GRAPH/Graph-Graph/data_preparation/global_features'

# Paths to YOLO files
cfg_path = '/Users/punsisikiridana/Documents/University_Education/MSc/SEMESTER_2/IN24_S2_CS5801_Advanced_AI/Project/GRAPH/Graph-Graph/data_preparation/yolov3.cfg'
weights_path = '/Users/punsisikiridana/Documents/University_Education/MSc/SEMESTER_2/IN24_S2_CS5801_Advanced_AI/Project/GRAPH/Graph-Graph/data_preparation/yolov3.weights'
names_path = '/Users/punsisikiridana/Documents/University_Education/MSc/SEMESTER_2/IN24_S2_CS5801_Advanced_AI/Project/GRAPH/Graph-Graph/data_preparation/coco.names'

#################### Creating directories ###########################
print("Creating directories...")
os.makedirs(img_output_dir, exist_ok=True)
os.makedirs(npz_output_dir, exist_ok=True)
os.makedirs(output_folder_dim, exist_ok=True)
os.makedirs(output_folder_global, exist_ok=True)
os.makedirs(npz_output_dir+"/Positive",exist_ok=True)
os.makedirs(npz_output_dir+"/Negative",exist_ok=True)
os.makedirs(output_folder_dim+"/Positive",exist_ok=True)
os.makedirs(output_folder_dim+"/Negative",exist_ok=True)
os.makedirs(output_folder_global+"/Positive",exist_ok=True)
os.makedirs(output_folder_global+"/Negative",exist_ok=True)

###################### Loading YOLO Model ######################
print("Loading YOLO Model...")
def load_yolo_model(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, 'r') as f:
        classes = f.read().strip().split('\n')
    return net, classes

# Load YOLO model
yolo_model, classes = load_yolo_model(cfg_path, weights_path, names_path)

####################### Detecting Objects ####################
print("Detecting objects...")
def detect_objects(net, classes, frame, conf_threshold=0.5, nms_threshold=0.4):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    h, w = frame.shape[:2]

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, x + width, y + height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    filtered_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            filtered_boxes.append(box + [confidences[i], class_ids[i]])

    return filtered_boxes

###################### Processing Video ######################
print("Generating object features...")
def process_video(video_path, yolo_model, classes, output_dir,npz_output_dir, accident=False, target_fps=20):
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    box_features_list = []
    detections = []
    labels = []
    frame_ids = []
    spatial_adj_matrices = []
    temporal_adj_matrices = []

    frame_count = 0
    saved_frame_indices = [0, 10, 20, 30, 40]  # Indices of frames to save with bounding boxes

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)
    frame_interval =1

    previous_frame_boxes = []
    previous_frame_features = []

    while cap.isOpened() and frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Extract frame-level features (using a CNN, here we just flatten the frame as a placeholder)
            frame_feature = cv2.resize(frame, (224, 224)).flatten()[:4096]  # Resize to ensure it fits 4096 elements

            frame_features.append(frame_feature)

            # Detect objects
            boxes = detect_objects(yolo_model, classes, frame)
            if len(boxes) > 19:
                boxes = boxes[:19]  # Ensure only 19 boxes are considered

            # Create box-level features (using the same placeholder approach for now)
            box_features = []
            for box in boxes:
                x1, y1, x2, y2, _, class_id = box
                box_frame = frame[y1:y2, x1:x2]
                if box_frame.size > 0:
                    box_feature = cv2.resize(box_frame, (64, 64)).flatten()[:4096]
                    box_features.append(box_feature)
                else:
                    box_features.append(np.zeros(4096))  # In case of invalid box

            while len(box_features) < 19:
                box_features.append(np.zeros(4096))  # Pad with zero features if less than 19 boxes

            box_features_list.append(box_features)
            # Convert boxes to a fixed size array
            while len(boxes) < 19:
                boxes.append([0, 0, 0, 0, 0.0, -1])  # Pad with default values if less than 19 boxes
            detections.append(np.array(boxes))

            # Create spatial adjacency matrix
            spatial_adj_matrix = np.zeros((len(boxes), len(boxes)))
            for i in range(len(boxes)):
                for j in range(len(boxes)):
                    if i != j:
                        ci = ((boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2)
                        cj = ((boxes[j][0] + boxes[j][2]) / 2, (boxes[j][1] + boxes[j][3]) / 2)
                        spatial_adj_matrix[i, j] = np.exp(-np.linalg.norm(np.array(ci) - np.array(cj)))
            spatial_adj_matrices.append(spatial_adj_matrix)

            # Create temporal adjacency matrix
            if len(previous_frame_boxes) > 0:
                temporal_adj_matrix = np.zeros((len(boxes), len(previous_frame_boxes)))
                for i in range(len(boxes)):
                    for j in range(len(previous_frame_boxes)):
                        if classes[int(boxes[i][5])] == classes[int(previous_frame_boxes[j][5])]:
                            temporal_adj_matrix[i, j] = cosine_similarity(
                                [box_features[i]], [previous_frame_features[j]]
                            )[0, 0]
            else:
                temporal_adj_matrix = np.zeros((len(boxes), 19))  # Use a default zero matrix if no previous frame

            temporal_adj_matrices.append(temporal_adj_matrix)

            # Save previous frame boxes and features
            previous_frame_boxes = boxes
            previous_frame_features = box_features

            # Save frames with bounding boxes
            if len(frame_features) - 1 in saved_frame_indices:
                for box in boxes:
                    x1, y1, x2, y2, confidence, class_id = box
                    label = f"{classes[int(class_id)]}: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                rnum = random.randint(0,999999999999999)
                output_frame_path = os.path.join(output_dir, f"frame_{len(frame_features) - 1}{str(rnum)}.jpg")
                cv2.imwrite(output_frame_path, frame)

            # Assign a label (placeholder, replace with actual logic)
            if accident:
                label = [0, 1]
            else:
                label = [1, 0]

            # Collect frame IDs (placeholder, replace with actual video name logic)
            frame_id = video_path.split('/')[-1].split('.')[0].strip()
            
        frame_count += 1

    cap.release()

    # Ensure we have exactly 50 frames
    while len(frame_features) < 100:
        frame_features.append(np.zeros(4096))
        detections.append(np.array([[0, 0, 0, 0, 0.0, -1]] * 19))
        box_features_list.append([np.zeros(4096)] * 19)
        spatial_adj_matrices.append(np.zeros((19, 19)))
        temporal_adj_matrices.append(np.zeros((19, 19)))

    # Convert lists to numpy arrays
    frame_features = np.array(frame_features).reshape((100, 1, 4096))
    box_features = np.array(box_features_list).reshape((100, 19, 4096))
    detections = np.array(detections, dtype='float16').reshape((100, 19, 6))
    label = np.array(label, dtype='float16')
    frame_ids = np.array(frame_ids, dtype='<U11')
    spatial_adj_matrices = np.array(spatial_adj_matrices).reshape((100, 19, 19))
    temporal_adj_matrices = np.array(temporal_adj_matrices).reshape((100, 19, 19))

    # Combine frame-level and box-level features
    combined_features = np.concatenate((frame_features, box_features), axis=1)

    final_output = combined_features.astype(np.float16)
    if accident:
        np.savez('{}/Positive/{}.npz'.format(npz_output_dir,frame_id), data=final_output, det=detections, labels=label, ID=frame_id)
        print("NPZ file successfully created : {}/Positive/{}.npz".format(npz_output_dir,frame_id))
    else:
        np.savez('{}/Negative/{}.npz'.format(npz_output_dir,frame_id), data=final_output, det=detections, labels=label, ID=frame_id)
        print("NPZ file successfully created : {}/Negative/{}.npz".format(npz_output_dir,frame_id))
    
    return final_output, detections, label, frame_ids

################################ Executing the process #################################

def execute_process(input_folder_path, yolo_model, classes, output_dir,npz_output_dir,accident=False):
    for file_name in os.listdir(input_folder_path):
        if file_name.endswith('.mp4'):
            video_path = os.path.join(input_folder_path, file_name)
            data, det, labels, ID = process_video(video_path, yolo_model, classes, output_dir, npz_output_dir, accident, target_fps=20)
            print(f"Processed video: {file_name}")
            print("data shape: ", data.shape)
            print("det shape: ", det.shape)
            print("labels shape: ", labels.shape)
            print("ID shape: ", ID.shape)


#accident=False
#execute_process(input_folder_negative, yolo_model, classes, img_output_dir,npz_output_dir,accident)

accident = True
#execute_process(input_folder_positive, yolo_model, classes, img_output_dir,npz_output_dir,accident)
################################# Generate & save video dimensions #################################
print("Generating frames stats...")
def get_video_dimensions(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None
    
    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Release the video capture object
    cap.release()
    
    return width, height

def save_video_dimensions(folder_path, output_folder,accident):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only .mp4 files
    mp4_files = [file for file in files if file.endswith('.mp4')]
    
    # Ensure the output folder exists
    #os.makedirs(output_folder, exist_ok=True)
    
    # Get dimensions for each video file and save as .npy
    for file_name in mp4_files:
        video_path = os.path.join(folder_path, file_name)
        dims = get_video_dimensions(video_path)
        if dims:
            # Create an array with 100 [width, height] entries
            repeated_dims = np.tile(dims, (100, 1))
            # Save the NumPy array as a .npy file
            if accident:
                output_path = os.path.join(output_folder+"/Positive/", f"{os.path.splitext(file_name)[0]}.npy")
            else:
                output_path = os.path.join(output_folder+"/Negative/", f"{os.path.splitext(file_name)[0]}.npy")
            np.save(output_path, repeated_dims)
            #print(len(repeated_dims))
            #print(repeated_dims)
            print(f"Dimensions array for {file_name} saved to {output_path}")

# Example usage
accident = True
save_video_dimensions(input_folder_positive, output_folder_dim,accident)
#accident = False
#save_video_dimensions(input_folder_negative,output_folder_dim,accident)


#################################### Extracting global features #################################
print("Extracting global features...")
# Load the pre-trained VGG16 model
vgg16 = models.vgg16(pretrained=True)

# Modify the model to remove the classification layers and retain feature extraction layers
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, vgg16):
        super(VGG16FeatureExtractor, self).__init__()
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.fc = nn.Sequential(
            nn.Linear(25088, 4096),  # VGG16 has a 4096-dim output from the first FC layer
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),  # Reduce to 2048-dim
            nn.ReLU(True),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Instantiate the feature extractor
feature_extractor = VGG16FeatureExtractor(vgg16)
feature_extractor.eval()

# Function to extract frames from a video
def extract_frames(video_path, num_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, frame_count // num_frames)  # Ensure interval is at least 1
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3)))  # Pad with black frames if fewer frames are available

    cap.release()
    frames = np.array(frames)
    return frames

# Function to extract global features using VGG16
def extract_global_features(video_path, output_npy='global_features.npy'):
    # Extract frames
    frames = extract_frames(video_path, num_frames=100)

    # Preprocess the frames
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    frames = torch.stack([preprocess(frame) for frame in frames])  # Stack into a tensor
    frames = frames.unsqueeze(0)  # Add batch dimension, shape (1, 100, 3, 224, 224)
    
    # Flatten batch and temporal dimension for VGG16
    frames = frames.view(-1, 3, 224, 224)  # Shape will be (100, 3, 224, 224)

    # Ensure the input is of type float32 to match the model's parameters
    frames = frames.float()

    # Pass through the VGG16 feature extractor
    with torch.no_grad():
        features = feature_extractor(frames)

    features = features.view(100, -1)  # Ensure shape is (100, 2048)
    features = features.numpy()

    # Save to a .npy file
    np.save(output_npy, features)
    print(f"Features saved to {output_npy}")


def process_global_features(input_folder,output_folder,accident):
    # Iterate through the videos in the input folder
    for video_file in os.listdir(input_folder):
        if video_file.endswith('.mp4') or video_file.endswith('.avi'):
            video_path = os.path.join(input_folder, video_file)
            if accident:
                output_npy = os.path.join(output_folder + '/Positive/', f'{os.path.splitext(video_file)[0]}.npy')
            else:
                output_npy = os.path.join(output_folder+'/Negative/', f'{os.path.splitext(video_file)[0]}.npy')
            extract_global_features(video_path, output_npy=output_npy)

# Process the global features for positive and negative cases
accident = True
#process_global_features(input_folder_positive,output_folder_global,accident)
#accident = False
#process_global_features(input_folder_negative,output_folder_global,accident)

end_time = time.time() 
run_time = end_time - start_time 
print(f"Script run time: {run_time} seconds")
################################### END #####################################
