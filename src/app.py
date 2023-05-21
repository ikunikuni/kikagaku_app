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

def gen(camera): #f
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed') #f
def video_feed():
    return Response(gen.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')



#ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#def allowed_file(filename):
    #return ('.') in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




        #ret, buffer = cv2.imencode('.jpg', frame) #?
        #frame = buffer.tobytes() #?
        #frame_bytes = buffer.tobytes() #?

        #yield (b'--frame\r\n' #?
               #b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') #?
               #b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n') #?

    #cap.release()


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        image_data_url = request.form['imageData'] #cam
        image_data = re.sub('^data:image/.+;base64,', '', image_data_url) #cam
        image = Image.open(BytesIO(base64.b64decode(image_data))) #cam

        pred = predict(image)
        seibetsuDanjyo_ = getDanjyo(pred)
        return render_template('result.html', seibetsuDanjyo=seibetsuDanjyo_, image=image_data_url) #cam

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

#if __name__ == '__main__':
    #app.run(host='0.0.0.0', debug=True)