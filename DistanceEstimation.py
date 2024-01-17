import cv2 as cv 
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

previous_positions = {}
recent_angles = {}  # Store recent angles
ANGLE_AVERAGE_FRAME_COUNT = 5  # Number of frames to average over
recent_positions = {}  # Store recent positions
POSITION_AVERAGE_FRAME_COUNT = 5  # Number of frames to average over

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
        print("classid:", classid)
        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        center_x, center_y = box[0] + box[2] // 2, box[1] + box[3] // 2
        data_list.append([class_names[classid], box[2], (box[0], box[1]-2), (center_x, center_y)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

def get_direction(angle):
    if angle is None:
        return "Stationary"
    elif 45 < angle <= 135:
        return "Moving Up"
    elif 135 < angle <= 225:
        return "Moving Left"
    elif 225 < angle <= 315:
        return "Moving Down"
    else:  # Covers angles from 315 to 45
        return "Moving Right"

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    data = object_detector(frame) 
    for d in data:
        class_name, _, _, center = d[0], d[1], d[2], d[3]

        # Initialize avg_x, avg_y, and avg_angle with default values
        avg_x, avg_y = center
        avg_angle = None

        # Before calculating average position, check if the class_name exists in recent_positions
        if class_name in recent_positions and len(recent_positions[class_name]) > 0:
            # Calculate average position
            avg_x = sum(pos[0] for pos in recent_positions[class_name]) / len(recent_positions[class_name])
            avg_y = sum(pos[1] for pos in recent_positions[class_name]) / len(recent_positions[class_name])

            # Rest of your code for calculating average angle...
        else:
            # Handle the case where recent_positions does not have any entry for the class_name
            # You can choose to print a warning, initialize the position, or take other appropriate actions
            print(f"No recent positions recorded for class {class_name}")

        # Calculate average position if class_name is in recent_positions
        if class_name in recent_positions:
            recent_positions[class_name].append(center)
            if len(recent_positions[class_name]) > POSITION_AVERAGE_FRAME_COUNT:
                recent_positions[class_name].pop(0)

            # Calculate average position
        avg_x = sum(pos[0] for pos in recent_positions[class_name]) / len(recent_positions[class_name])
        avg_y = sum(pos[1] for pos in recent_positions[class_name]) / len(recent_positions[class_name])

    # Calculate average angle if class_name is in previous_positions
    if class_name in previous_positions:
        prev_center = previous_positions[class_name]
        delta_x, delta_y = avg_x - prev_center[0], avg_y - prev_center[1]

        MOVEMENT_THRESHOLD = 10  # Minimum number of pixels the object must move to be considered actual movement

        if abs(delta_x) > MOVEMENT_THRESHOLD or abs(delta_y) > MOVEMENT_THRESHOLD:
            angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)
            if angle < 0:
                angle += 360

            if class_name not in recent_angles:
                recent_angles[class_name] = [angle]
            else:
                recent_angles[class_name].append(angle)
                if len(recent_angles[class_name]) > ANGLE_AVERAGE_FRAME_COUNT:
                    recent_angles[class_name].pop(0)

                # Calculate average angle
                avg_angle = sum(recent_angles[class_name]) / len(recent_angles[class_name])
        else:
            recent_positions[class_name] = []  # Reset position list if object is not detected

        previous_positions[class_name] = (avg_x, avg_y)

        # Use avg_angle instead of angle for direction calculation
        direction = get_direction(avg_angle)
        
        # Display direction
        angle_text = f'Angle: {angle:.2f} degrees' if angle is not None else "Angle: N/A"
        direction_text = f'{angle_text}, {direction}'
        cv.putText(frame, direction_text, (x, y-25), FONTS, 0.5, (0, 0, 255), 2)  # Red color for text

    cv.imshow('frame', frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
