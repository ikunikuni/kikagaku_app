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

def predict(img):
    net = Net().cpu().eval()
    net.load_state_dict(torch.load('seibetsujudge_cpu.pt', map_location=torch.device('cpu')))

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

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

def allowed_file(filename):
    return ('.') in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        #if 'filename' not in request.files:
            #return redirect(request.url)
        #file = request.files['filename']
        #if file and allowed_file(file.filename):
            #buf = io.BytesIO()
            #image = Image.open(file)
            #image.save(buf, 'png')
            #base64_str = base64.b64encode(buf.getvalue()).decode('UTF-8')
            #base64_data = 'data:image/png;base64,{}'.format(base64_str)

        image_data_url = request.form['imageData']
        image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        pred = predict(image)
        seibetsuDanjyo_ = getDanjyo(pred)
        #return render_template('result.html', seibetsuDanjyo=seibetsuDanjyo_, image=base64_data)
        return render_template('result.html', seibetsuDanjyo=seibetsuDanjyo_, image=image_data_url)
        return redirect(request.url)

    elif request.method == 'GET':
        return render_template('webcam.html')
    


def detect_gender(txt):
    if '男' in txt:
        gender = '男'
    elif '女' in txt:
        gender = '女'
    else:
        gender = '不明'
    return gender


def detect_birthdate(txt):
    birthdate_regex = r'(平成|昭和|大正|明治|令和)(\d+|[元一二三四五六七八九十]+)年\s+(\d+月\d+日)生'
    birthdate_match = re.search(birthdate_regex, txt)
    if birthdate_match:
        era = birthdate_match.group(1)
        year = birthdate_match.group(2)
        if year == "元":
            year = "1"

        if era == "平成":
            year_number = int(year) + 1988
            date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        elif era == "昭和":
            year_number = int(year) + 1925
            date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        elif era == "大正":
            year_number = int(year) + 1911
            date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        elif era == "明治":
            year_number = int(year) + 1867
            date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        elif era == "令和":
            year_number = int(year) + 2018
            date = str(year_number) + '年' + birthdate_match.group(3)  # 年数 + 年月日
        else:
            date = era + year + '年' + birthdate_match.group(3)  # 元号 + 年数 + 年月日
    
        
        birth_day = datetime.datetime.strptime(date, '%Y年%m月%d日')
        current_date = datetime.datetime.now()
        age = current_date.year - birth_day.year - ((current_date.month, current_date.day) < (birth_day.month, birth_day.day))
        return age
    else:
        return None  


# OCRツールの初期化
tools = pyocr.get_available_tools()
tool = tools[0]

@app.route('/mynum', methods=['GET', 'POST'])
def mynum():
    if request.method == 'POST':
        # リクエストから画像ファイルを取得
        file2 = request.files['filename2']
        # PIL.Imageに変換
        img2 = Image.open(file2)
        # 画像からテキストを抽出
        txt = tool.image_to_string(img2, lang='jpn+eng',builder=pyocr.builders.TextBuilder(tesseract_layout=6))
        # 抽出したテキストから性別を判定
        gender = detect_gender(txt)
        if gender == '男':
            gender = 'male'
        elif gender == '女':
            gender = 'female'
        else:
            gender = 'unknown'
        # 抽出したテキストから年齢を判定
        age = detect_birthdate(txt)
        # 結果をレンダリングするテンプレートに渡してレスポンスを返す
        return render_template('result_mynum.html', gender=gender, age=age)
    else:
        return render_template('mynum.html')







if __name__ == '__main__':
    app.run(debug=True)