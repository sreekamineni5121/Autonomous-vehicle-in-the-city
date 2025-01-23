
from keras.applications.mobilenet_v2 import MobileNetV2

class ObjectDetection:
    def __init__(self):
        self.object_detection_model = MobileNetV2(weights='imagenet')
    
    def detect_objects(self, frame):
        objects = self.object_detection_model.predict(np.expand_dims(cv2.resize(frame, (224, 224)), axis=0))
        return objects
