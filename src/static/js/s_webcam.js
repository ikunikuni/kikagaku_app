var video = document.getElementById('video');
var captureButton = document.getElementById('capture');
var imageDataInput = document.getElementById('imageData');
var segmentedCanvas = document.getElementById('segmentedCanvas');

// カメラのストリームを取得してビデオ要素に表示する関数
function startVideo() {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (error) {
      console.error('カメラのストリームを取得できませんでした: ', error);
    });
}

// カメラから静止画をキャプチャし格納する関数
function captureImage(canvas) {
  var imageData = canvas.toDataURL('image/png');
  imageDataInput.value = imageData;
}


// セグメンテーション後の画像を指定のキャンバスに描画する関数
function displaySegmentedImage(segmentedCanvas, imageData) {
  var context = segmentedCanvas.getContext('2d');
  var image = new Image();
  image.onload = function () {
    context.clearRect(0, 0, segmentedCanvas.width, segmentedCanvas.height);
    context.drawImage(image, 0, 0, segmentedCanvas.width, segmentedCanvas.height);
  };
  image.src = imageData;
}

// 撮影ボタンのクリックイベントハンドラ
captureButton.addEventListener('click', function () {
  // オリジナル画像のキャプチャ
  captureImage(canvas);

  // セグメンテーション後の処理
  // performSegmentation()関数でセグメンテーションを実行し、
  // セグメンテーション後の画像データをsegmentedImageDataに取得する
  var segmentedImageData = performSegmentation(canvas);

  // セグメンテーション後の画像を表示する
  displaySegmentedImage(segmentedCanvas, segmentedImageData);
});

// ページ読み込み時にカメラのストリームを開始する
window.addEventListener('load', function () {
  startVideo();
});
