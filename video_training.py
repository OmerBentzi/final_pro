
from super_gradients.training import models
import torch
import cv2
import numpy as np
import pandas

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# start webcam
cap = cv2.VideoCapture('my.mp4')
cap.set(3, 640)
cap.set(4, 480)
model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose" )
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
confidence = 0.6
results_df = pandas.DataFrame([], columns=['right_elbow_right_shoulder_right_hip', 'left_elbow_left_shoulder_left_hip', 'right_knee_mid_hip_left_knee', 'right_hip_right_knee_right_ankle', 'left_hip_left_knee_left_ankle', 'right_wrist_right_elbow_right_shoulder', 'left_wrist_left_elbow_left_shoulder' , 'x_left_shoulder_left_wrist',
       'y_left_shoulder_left_wrist', 'z_left_shoulder_left_wrist',
       'x_right_shoulder_right_wrist', 'y_right_shoulder_right_wrist',
       'z_right_shoulder_right_wrist', 'x_left_hip_left_ankle',
       'y_left_hip_left_ankle', 'z_left_hip_left_ankle',
       'x_right_hip_right_ankle', 'y_right_hip_right_ankle',
       'z_right_hip_right_ankle', 'x_left_hip_left_wrist',
       'y_left_hip_left_wrist', 'z_left_hip_left_wrist',
       'x_right_hip_right_wrist', 'y_right_hip_right_wrist',
       'z_right_hip_right_wrist', 'x_left_shoulder_left_ankle',
       'y_left_shoulder_left_ankle', 'z_left_shoulder_left_ankle',
       'x_right_shoulder_right_ankle', 'y_right_shoulder_right_ankle',
       'z_right_shoulder_right_ankle', 'x_left_hip_right_wrist',
       'y_left_hip_right_wrist', 'z_left_hip_right_wrist',
       'x_right_hip_left_wrist', 'y_right_hip_left_wrist',
       'z_right_hip_left_wrist', 'x_left_elbow_right_elbow',
       'y_left_elbow_right_elbow', 'z_left_elbow_right_elbow',
       'x_left_knee_right_knee', 'y_left_knee_right_knee',
       'z_left_knee_right_knee', 'x_left_wrist_right_wrist',
       'y_left_wrist_right_wrist', 'z_left_wrist_right_wrist',
       'x_left_ankle_right_ankle', 'y_left_ankle_right_ankle',
       'z_left_ankle_right_ankle'])

while(cap.isOpened()):
    success, img = cap.read()

    if success:

        
        results = model.predict(img, conf=confidence ,fuse_model=False)
        prediction = results.prediction
        poses = prediction.poses
        edge_links = prediction.edge_links
        if len(poses) > 0:
            for pose in range(len(poses)):
                for link in edge_links:
                    try:
                        point1 = poses[pose][link[0]]
                        point2 = poses[pose][link[1]]

                        x1, y1 = int(point1[0]), int(point1[1])
                        x2, y2 = int(point2[0]), int(point2[1])

                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    except:
                     print("no human")
            cv2.imshow('Frame', img)
            for pose in poses:
                left_shoulder = pose[5]
                left_wrist = pose[9]
                right_shoulder = pose[6]
                right_wrist = pose[10]
                left_hip = pose[11]
                left_ankle = pose[15]
                right_hip = pose[12]
                right_ankle = pose[16]
                left_elbow = pose[7]
                right_elbow = pose[8]
                left_knee = pose[13]
                right_knee = pose[14]

                mid_hip = [(right_hip[0] + left_hip[0])/2 , (right_hip[1] + left_hip[1])/2 , (right_hip[2] + left_hip[2])/2]
                # Calculate angles
                left_elbow_left_shoulder_left_hip = calculate_angle(left_elbow, left_shoulder, left_hip)
                right_knee_mid_hip_left_knee = calculate_angle(right_knee, mid_hip, left_knee)
                right_hip_right_knee_right_ankle = calculate_angle(right_hip, right_knee, right_ankle)
                left_hip_left_knee_left_ankle = calculate_angle(left_hip, left_knee, left_ankle)
                right_wrist_right_elbow_right_shoulder = calculate_angle(right_wrist, right_elbow, right_shoulder)
                left_wrist_left_elbow_left_shoulder = calculate_angle(left_wrist, left_elbow, left_shoulder)
                right_elbow_right_shoulder_right_hip = calculate_angle(right_elbow, right_shoulder, right_hip)

                x_left_shoulder_left_wrist = left_wrist[0]-left_shoulder[0]  
                y_left_shoulder_left_wrist = left_wrist[1]-left_shoulder[1]  
                z_left_shoulder_left_wrist = left_wrist[2]-left_shoulder[2]  
                x_right_shoulder_right_wrist = right_wrist[0]-right_shoulder[0]  
                y_right_shoulder_right_wrist = right_wrist[1]-right_shoulder[1]  
                z_right_shoulder_right_wrist = right_wrist[2]-right_shoulder[2]  
                x_left_hip_left_ankle = left_ankle[0]-left_hip[0]  
                y_left_hip_left_ankle = left_ankle[1]-left_hip[1]   
                z_left_hip_left_ankle = left_ankle[2]-left_hip[2]  
                x_right_hip_right_ankle = right_ankle[0]-right_hip[0]  
                y_right_hip_right_ankle = right_ankle[1]-right_hip[1]  
                z_right_hip_right_ankle = right_ankle[2]-right_hip[2]  
                x_left_hip_left_wrist = left_wrist[0]-left_hip[0]  
                y_left_hip_left_wrist = left_wrist[1]-left_hip[1]  
                z_left_hip_left_wrist = left_wrist[2]-left_hip[2]  
                x_right_hip_right_wrist = right_wrist[0]-right_hip[0]  
                y_right_hip_right_wrist = right_wrist[1]-right_hip[1]  
                z_right_hip_right_wrist = right_wrist[2]-right_hip[2]  
                x_left_shoulder_left_ankle = left_ankle[0]-left_shoulder[0]  
                y_left_shoulder_left_ankle = left_ankle[1]-left_shoulder[1]  
                z_left_shoulder_left_ankle = left_ankle[2]-left_shoulder[2]  
                x_right_shoulder_right_ankle = right_ankle[0]-right_shoulder[0]  
                y_right_shoulder_right_ankle = right_ankle[1]-right_shoulder[1]  
                z_right_shoulder_right_ankle = right_ankle[2]-right_shoulder[2]  
                x_left_hip_right_wrist = right_wrist[0]-left_hip[0]  
                y_left_hip_right_wrist = right_wrist[1]-left_hip[1]  
                z_left_hip_right_wrist = right_wrist[2]-left_hip[2]  
                x_right_hip_left_wrist = left_wrist[0]-right_hip[0]  
                y_right_hip_left_wrist = left_wrist[1]-right_hip[1]  
                z_right_hip_left_wrist = left_wrist[2]-right_hip[2]  
                x_left_elbow_right_elbow = right_elbow[0]-left_elbow[0]  
                y_left_elbow_right_elbow = right_elbow[1]-left_elbow[1]  
                z_left_elbow_right_elbow = right_elbow[2]-left_elbow[2]  
                x_left_knee_right_knee = right_knee[0]-left_knee[0]  
                y_left_knee_right_knee = right_knee[1]-left_knee[1]  
                z_left_knee_right_knee = right_knee[2]-left_knee[2]  
                x_left_wrist_right_wrist = right_wrist[0]-left_wrist[0]  
                y_left_wrist_right_wrist = right_wrist[1]-left_wrist[1]  
                z_left_wrist_right_wrist = right_wrist[2]-left_wrist[2]   
                x_left_ankle_right_ankle = right_ankle[0]-left_ankle[0]  
                y_left_ankle_right_ankle = right_ankle[1]-left_ankle[1]  
                z_left_ankle_right_ankle = right_ankle[2]-left_ankle[2]  

                print("left_elbow_left_shoulder_left_hip" , left_elbow_left_shoulder_left_hip)
                print("right_knee_mid_hip_left_knee" , right_knee_mid_hip_left_knee)
                print("right_hip_right_knee_right_ankle" , right_hip_right_knee_right_ankle)
                print("left_hip_left_knee_left_ankle" , left_hip_left_knee_left_ankle)
                print("right_wrist_right_elbow_right_shoulder" , right_wrist_right_elbow_right_shoulder)
                print("left_wrist_left_elbow_left_shoulder" , left_wrist_left_elbow_left_shoulder)
                print("right_elbow_right_shoulder_right_hip" , right_elbow_right_shoulder_right_hip)
                print("x_left_shoulder_left_wrist" , x_left_shoulder_left_wrist)
            angles = [right_elbow_right_shoulder_right_hip, left_elbow_left_shoulder_left_hip, right_knee_mid_hip_left_knee, right_hip_right_knee_right_ankle, left_hip_left_knee_left_ankle, right_wrist_right_elbow_right_shoulder, left_wrist_left_elbow_left_shoulder , x_left_shoulder_left_wrist,
       y_left_shoulder_left_wrist, z_left_shoulder_left_wrist,
       x_right_shoulder_right_wrist, y_right_shoulder_right_wrist,
       z_right_shoulder_right_wrist, x_left_hip_left_ankle,
       y_left_hip_left_ankle, z_left_hip_left_ankle,
       x_right_hip_right_ankle, y_right_hip_right_ankle,
       z_right_hip_right_ankle, x_left_hip_left_wrist,
       y_left_hip_left_wrist, z_left_hip_left_wrist,
       x_right_hip_right_wrist, y_right_hip_right_wrist,
       z_right_hip_right_wrist, x_left_shoulder_left_ankle,
       y_left_shoulder_left_ankle, z_left_shoulder_left_ankle,
       x_right_shoulder_right_ankle, y_right_shoulder_right_ankle,
       z_right_shoulder_right_ankle, x_left_hip_right_wrist,
       y_left_hip_right_wrist, z_left_hip_right_wrist,
       x_right_hip_left_wrist, y_right_hip_left_wrist,
       z_right_hip_left_wrist, x_left_elbow_right_elbow,
       y_left_elbow_right_elbow, z_left_elbow_right_elbow,
       x_left_knee_right_knee, y_left_knee_right_knee,
       z_left_knee_right_knee, x_left_wrist_right_wrist,
       y_left_wrist_right_wrist, z_left_wrist_right_wrist,
       x_left_ankle_right_ankle, y_left_ankle_right_ankle,
       z_left_ankle_right_ankle] 
       
            angles_df = pandas.DataFrame([angles], columns=['right_elbow_right_shoulder_right_hip', 'left_elbow_left_shoulder_left_hip', 'right_knee_mid_hip_left_knee', 'right_hip_right_knee_right_ankle', 'left_hip_left_knee_left_ankle', 'right_wrist_right_elbow_right_shoulder', 'left_wrist_left_elbow_left_shoulder' , 'x_left_shoulder_left_wrist',
       'y_left_shoulder_left_wrist', 'z_left_shoulder_left_wrist',
       'x_right_shoulder_right_wrist', 'y_right_shoulder_right_wrist',
       'z_right_shoulder_right_wrist', 'x_left_hip_left_ankle',
       'y_left_hip_left_ankle', 'z_left_hip_left_ankle',
       'x_right_hip_right_ankle', 'y_right_hip_right_ankle',
       'z_right_hip_right_ankle', 'x_left_hip_left_wrist',
       'y_left_hip_left_wrist', 'z_left_hip_left_wrist',
       'x_right_hip_right_wrist', 'y_right_hip_right_wrist',
       'z_right_hip_right_wrist', 'x_left_shoulder_left_ankle',
       'y_left_shoulder_left_ankle', 'z_left_shoulder_left_ankle',
       'x_right_shoulder_right_ankle', 'y_right_shoulder_right_ankle',
       'z_right_shoulder_right_ankle', 'x_left_hip_right_wrist',
       'y_left_hip_right_wrist', 'z_left_hip_right_wrist',
       'x_right_hip_left_wrist', 'y_right_hip_left_wrist',
       'z_right_hip_left_wrist', 'x_left_elbow_right_elbow',
       'y_left_elbow_right_elbow', 'z_left_elbow_right_elbow',
       'x_left_knee_right_knee', 'y_left_knee_right_knee',
       'z_left_knee_right_knee', 'x_left_wrist_right_wrist',
       'y_left_wrist_right_wrist', 'z_left_wrist_right_wrist',
       'x_left_ankle_right_ankle', 'y_left_ankle_right_ankle',
       'z_left_ankle_right_ankle'])
        
            results_df = pandas.concat([results_df , angles_df])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # If no more frames, break the loop
        break

cap.release()
cv2.destroyAllWindows()
results_df.to_csv('results_df1.csv', index=False)