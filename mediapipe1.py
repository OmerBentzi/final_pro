import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_distances
from collections import deque , Counter
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

@app.route('/predict')
def predict():
    # Your route logic here
    pass


app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'class': clf.predict([request.json['angles']])[0]})



mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

merged_df2 = pandas.read_csv('results_finaly_test.csv')
X = merged_df2.drop(["class"], axis = "columns").values
Y = merged_df2['class']
clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
clf.fit(X, Y)
cap = cv2.VideoCapture(0)
frames = deque(maxlen=50)
results_df = pandas.DataFrame([], columns=['right_elbow_right_shoulder_right_hip', 'left_elbow_left_shoulder_left_hip', 'right_knee_mid_hip_left_knee', 'right_hip_right_knee_right_ankle', 'left_hip_left_knee_left_ankle', 'right_wrist_right_elbow_right_shoulder', 'left_wrist_left_elbow_left_shoulder'])
confidence = 0.6

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]

            # Calculate angle
            mid_hip = [(right_hip[0] + left_hip[0])/2 , (right_hip[1] + left_hip[1])/2 , (right_hip[2] + left_hip[2])/2]
            angle_left_shoulder_left_elbow_left_wrist = calculate_angle(left_shoulder, left_elbow, left_wrist)
            angle_right_shoulder_right_elbow_right_wrist = calculate_angle(right_shoulder, right_elbow, right_wrist)
            angle_left_elbow_left_shoulder_left_hip = calculate_angle(left_elbow, left_shoulder, left_hip)
            angle_right_knee_mid_hip_left_knee = calculate_angle(right_knee, mid_hip, left_knee)
            angle_right_hip_right_knee_right_ankle = calculate_angle(right_hip, right_knee, right_ankle)
            angle_left_hip_left_knee_left_ankle = calculate_angle(left_hip, left_knee, left_ankle)
            angle_right_elbow_right_shoulder_right_hip = calculate_angle(right_elbow, right_shoulder, right_hip) 

            angle_right_shoulder_right_hip_right_knee = calculate_angle(right_shoulder, right_hip, right_knee) 
            angle_left_shoulder_left_hip_left_knee = calculate_angle(left_shoulder, left_hip, left_knee) 
            
            angles_for_image = [angle_left_shoulder_left_elbow_left_wrist , angle_right_shoulder_right_elbow_right_wrist , angle_left_hip_left_knee_left_ankle , angle_right_hip_right_knee_right_ankle]
            locations_for_image = [left_elbow[:2] , right_elbow[:2] , left_knee[:2] , right_knee[:2]]
            for i in range(len(angles_for_image)):
                cv2.putText(image, str(int(angles_for_image[i])), 
                           tuple(np.multiply(locations_for_image[i], [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA
                                )     

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

            angles = [angle_right_elbow_right_shoulder_right_hip, angle_left_elbow_left_shoulder_left_hip, angle_right_knee_mid_hip_left_knee, angle_right_hip_right_knee_right_ankle, angle_left_hip_left_knee_left_ankle, angle_right_shoulder_right_elbow_right_wrist, angle_left_shoulder_left_elbow_left_wrist , x_left_shoulder_left_wrist,
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
            print(clf.predict([angles]))
            probs = clf.predict_proba([angles])[0]
            max_prob = max(probs)
            if max_prob < 0.8:
                print("no exercise")
                frames.append("no")
                exrcise = Counter(frames).most_common(1)[0][0]
                if exrcise == "no":
                    results_df = pandas.DataFrame([], columns=['right_elbow_right_shoulder_right_hip', 'left_elbow_left_shoulder_left_hip', 'right_knee_mid_hip_left_knee', 'right_hip_right_knee_right_ankle', 'left_hip_left_knee_left_ankle', 'right_wrist_right_elbow_right_shoulder', 'left_wrist_left_elbow_left_shoulder'])
                cv2.putText(image, f"E: {exrcise}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            else:
                # Find the peaks in the array of angles
                ang_df = pandas.DataFrame([[angle_right_elbow_right_shoulder_right_hip, angle_left_elbow_left_shoulder_left_hip, angle_right_knee_mid_hip_left_knee, angle_right_hip_right_knee_right_ankle, angle_left_hip_left_knee_left_ankle, angle_right_shoulder_right_elbow_right_wrist, angle_left_shoulder_left_elbow_left_wrist , angle_right_shoulder_right_hip_right_knee , angle_left_shoulder_left_hip_left_knee]], columns=['right_elbow_right_shoulder_right_hip', 'left_elbow_left_shoulder_left_hip', 'right_knee_mid_hip_left_knee', 'right_hip_right_knee_right_ankle', 'left_hip_left_knee_left_ankle', 'right_wrist_right_elbow_right_shoulder', 'left_wrist_left_elbow_left_shoulder' , 'angle_right_shoulder_right_hip_right_knee' , 'angle_left_shoulder_left_hip_left_knee'])
                results_df = pandas.concat([results_df , ang_df])
                max_index = np.argmax(probs)
                frames.append(clf.classes_[max_index])
                exrcise = Counter(frames).most_common(1)[0][0]
                if(clf.classes_[max_index] == 'squat'): 
                    peaks, _ = find_peaks(results_df['right_knee_mid_hip_left_knee'].astype(int) ,height=100, distance=10 , prominence=20)
                    # The number of repetitions is the number of peaks
                    repetitions = len(peaks)
                elif clf.classes_[max_index] == 'pushup':
                    peaks, _ = find_peaks(results_df['left_wrist_left_elbow_left_shoulder'].astype(int) ,height=65, distance=10 , prominence=20)
                    peaks2, _ = find_peaks(results_df['right_wrist_right_elbow_right_shoulder'].astype(int) ,height=65, distance=10 , prominence=20)
                    # The number of repetitions is the number of peaks
                    repetitions = max(len(peaks) ,len(peaks2))
                elif clf.classes_[max_index] == 'situp':
                    peaks, _ = find_peaks(results_df['angle_right_shoulder_right_hip_right_knee'].astype(int) ,height=50, distance=30 , prominence=20)
                    peaks2, _ = find_peaks(results_df['angle_left_shoulder_left_hip_left_knee'].astype(int) ,height=50, distance=30 , prominence=20)
                    # The number of repetitions is the number of peaks
                    repetitions = round((len(peaks) +len(peaks2))/2)
                # Print the exercise with the highest probability
                print(f"Exercise: {clf.classes_[max_index]}, Probability: {max_prob * 100}% , R:{repetitions}")
                
                # Calculate the cosine distance between the new observation and the existing ones
                distances = cosine_distances(angles_df.to_numpy().reshape(1, -1), X)

                # Convert the distance to a percentage
                accuracy = 100 * (1 - distances.min())
                cv2.putText(image, f"E: {clf.classes_[max_index]}, DEA: {int(max_prob * 100)}% , R:{repetitions} , A:{int(accuracy)}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                print(f"Accuracy: {accuracy}%")    

        except Exception as error:
            print(error)   

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    app.run(debug=True)