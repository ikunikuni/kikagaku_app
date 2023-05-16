import torch
from seibetsu import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64


def predict(img):
    net = Net().cpu().eval()
    net.load_state_dict(torch.load('seibetsujudge_cpu.pt', map_location=torch.device('cpu')))
    img = transform(img)
    img = img.unsqueeze(0)

    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y


def getDanjyo(label):
    if label == 0:
        return '女性'
    elif label ==1:
        return '男性'
    

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

def allwed_file(filename):
    return ('.') in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    if request.method == 'POST':
        if 'filename' not in request.files:
            return redirect(request.url)
        file = request.files['filename']
        if file and allwed_file(file.filename):

            buf = io.BytesIO()
            image = Image.open(file)
            image.save(buf, 'png')
            base64_str = base64.b64encode(buf.getvalue()).decode('UTF-8')
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            pred = predict(image)
            seibetsuDanjyo_ = getDanjyo(pred)
            return render_template('result.html', seibetsuDanjyo=seibetsuDanjyo_, image=base64_data)
        return redirect(request.url)

    
    elif request.method == 'GET':
        return render_template('index.html')
    

if __name__== '__main__':
    app.run(debug=True)



