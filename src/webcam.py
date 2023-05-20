import cv2

def detect_faces(frame):
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = detect_faces(frame)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()


    ret, frame = cv2.imencode('.jpg', frame)
    return frame.tobytes()

if __name__ == '__main__':
    main()
