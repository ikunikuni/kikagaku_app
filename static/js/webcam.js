const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const imageDataInput = document.getElementById('imageData');


const constraints = {
  video: true
};

navigator.mediaDevices.getUserMedia(constraints)
  .then((stream) => {
    video.srcObject = stream;
    video.play();
  });

captureButton.addEventListener('click', () => {
  const context = canvas.getContext('2d'); 
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageDataURL = canvas.toDataURL('image/jpeg', 1.0);
  imageDataInput.value = imageDataURL;
  form.submit();
});