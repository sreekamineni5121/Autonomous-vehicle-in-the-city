
import cv2
import threading
from advanced_traffic_sign_model import AdvancedTrafficSignModel
from object_detection import ObjectDetection
from lane_detection import LaneDetection

class SelfDrivingCar(AdvancedTrafficSignModel, ObjectDetection, LaneDetection):
    def __init__(self):
        AdvancedTrafficSignModel.__init__(self)
        ObjectDetection.__init__(self)
        self.cap = cv2.VideoCapture('self_driving_car_simulation.mp4')
        self.car_speed = 30
    
    def adjust_speed(self, traffic_sign):
        speed_limits = {0: 20, 1: 40, 2: 60, 3: 80}
        if traffic_sign in speed_limits:
            self.car_speed = speed_limits[traffic_sign]
        print(f'Adjusted Speed: {self.car_speed} km/h')
    
    def process_frame(self, frame):
        masked_edges = self.detect_lanes(frame)
        traffic_sign = self.predict_sign(frame)
        self.adjust_speed(traffic_sign)
        objects = self.detect_objects(frame)
        pedestrian_detected = any(obj for obj in objects if 'person' in str(obj))
        traffic_light_green = traffic_sign == 4
        turning_right = True
        
        if pedestrian_detected and (traffic_light_green or turning_right):
            self.car_speed = 0
            print("Pedestrian detected! Yielding...")
        
        cv2.putText(frame, f'Traffic Sign: {traffic_sign}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Speed: {self.car_speed} km/h', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Self-Driving Car Simulation', frame)
    
    def start(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_thread = threading.Thread(target=self.process_frame, args=(frame,))
            frame_thread.start()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    car = SelfDrivingCar()
    car.start()
