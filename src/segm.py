import cv2
import numpy as np
import base64
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# モデルのロード
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# 前処理
preprocess = transforms.Compose([
    transforms.Resize(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# imageDataから画像データを復元し、OpenCVのイメージ形式に変換する関数
def convert_image_data(segment_image):
    # base64デコードしてバイトデータに変換
    image_bytes = base64.b64decode(segment_image)
    # バイトデータをNumPy配列に変換
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # NumPy配列をOpenCVのイメージ形式に変換
    s_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return s_image

def perform_segmentation(s_image):
    # 画像の前処理
    s_img = Image.fromarray(s_image)
    img_tensor = preprocess(s_img)
    img_batch = img_tensor.unsqueeze(0)

    # 推論
    with torch.no_grad():
        output = model(img_batch)['out']

    # 推論結果の後処理
    out = torch.argmax(output[0], dim=0)
    mask = out.cpu().detach().numpy()

    # マスク画像をチャネル方向に 3 つ重ねる
    mask_rgb = np.stack([mask, mask, mask], axis=2)

    # 元の画像の読み込み
    origin = np.array(s_image)

    # 人物以外の箇所を黒色に置換
    result = np.where(mask_rgb == 15, origin, 0)

    #print(result)  # 結果をターミナルに表示

    return result
