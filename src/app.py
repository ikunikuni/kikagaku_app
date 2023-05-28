import torch
from seibetsu import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64
import cv2
from flask import Response
from torchvision import transforms
import pytesseract
from io import BytesIO
import pyocr
import re
import datetime
import easyocr
import numpy as np
from mynum import detect_gender
from mynum import detect_birthdate
import os
from flask import url_for



def predict(img):
    net = Net().cpu().eval()
    net.load_state_dict(torch.load('seibetsujudge_cpu_4.pt', map_location=torch.device('cpu')))

    # リサイズ
    transform = transforms.Compose([
        transforms.Resize(size=(100, 100)),
        transforms.ToTensor(),
    ])


    img = transform(img)
    img = img.unsqueeze(0)

    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

def getDanjyo(label):
    if label == 0:
        return '女性'
    elif label == 1:
        return '男性'

app = Flask(__name__)

camera = cv2.VideoCapture(0)

face_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        ret, frame = camera.read()  # フレームを取得
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        frame_with_faces = frame.copy()

        frame_with_faces = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
        

    # フレームをバイナリデータに変換
        ret, jpeg = cv2.imencode('.jpeg', frame_with_faces)
        frame_bytes = jpeg.tobytes()

    # フレームデータをジェネレータとして返す
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        image_data_url = request.form['imageData'] 
        image_data = re.sub('^data:image/.+;base64,', '', image_data_url) 
        image = Image.open(BytesIO(base64.b64decode(image_data))) 

        pred = predict(image)
        seibetsuDanjyo_ = getDanjyo(pred)
        return render_template('result.html', seibetsuDanjyo=seibetsuDanjyo_, image=image_data_url) 

    elif request.method == 'GET':
        return render_template('webcam.html')

    
#######################################################################################################


# OCRツールの初期化
reader = easyocr.Reader(['ja'])
tools = pyocr.get_available_tools()
tool = tools[0]

@app.route('/mynum', methods=['GET', 'POST'])
def mynum():
    if request.method == 'POST':
        # リクエストから画像ファイルを取得
        img_file = request.files['filename2']
        # OpenCVを使用して画像を読み込む
        img_bytes = img_file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_gen = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # EasyOCRを使用して性別を抽出
        img_gray = cv2.cvtColor(img_gen, cv2.COLOR_BGR2GRAY)
        txt_gender = reader.readtext(img_gray)



        gender = detect_gender(txt_gender)

        # PyOCRを使用して生年月日を抽出
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_pil = img_pil.convert('L')
        txt_age = tool.image_to_string(img_pil, lang='jpn+eng', builder=pyocr.builders.TextBuilder(tesseract_layout=6))

        age = detect_birthdate(txt_age)

        # 結果をレンダリングするテンプレートに渡してレスポンスを返す
        return render_template('result_mynum.html', gender=gender, age=age)
    else:
        return render_template('mynum.html')



if __name__ == '__main__':
    app.run(debug=True)
