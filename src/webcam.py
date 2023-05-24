import cv2

def detect_faces(frame):
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return frame

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)


    def __del__(self):
        self.video.release()

    #def get_frame(self):
        #success, f_image = self.video.read()
        #ret, jpeg = cv2.imencode('.jpg', f_image)
        #return jpeg.tobytes()
    
    def get_frame(self):
        success, frame = self.video.read()
        if success:
            frame_with_faces = detect_faces(frame)
            ret, jpeg = cv2.imencode('.jpg', frame_with_faces)
            return jpeg.tobytes()
        else:
            return None

