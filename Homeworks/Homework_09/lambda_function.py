import numpy as np
import onnxruntime
from PIL import Image
from io import BytesIO
from urllib import request
import os


MODEL_PATH = 'hair_classifier_empty.onnx'
TARGET_SIZE = (200, 200)

# --- INFERENCE/HANDLER FUNCTION ---
def download_and_preprocess(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    
    # 2. Prepare image (Resize and Convert)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(TARGET_SIZE, Image.Resampling.NEAREST)

    X = np.array(img, dtype=np.float32)
    X /= 255.0  
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    X = (X - mean) / std

    # 4. Add Batch dimension and Transpose (HWC -> CHW)
    X_input = X[np.newaxis, ...]           
    X_input = X_input.transpose(0, 3, 1, 2)
    
    return X_input.astype(np.float32)

def predict_local(url):
    X_input = download_and_preprocess(url)

    # Load the ONNX model 
    sess = onnxruntime.InferenceSession(MODEL_PATH)
    
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    result = sess.run([output_name], {input_name: X_input})
    prediction = result[0][0][0] 

    return prediction

# --- ENTRY POINT FOR DOCKER CMD ---
if __name__ == '__main__':
   
    image_url = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'
    
    print("Starting prediction using model inside Docker...")
    output = predict_local(image_url)
    
    print(f"\nQ6 Final Output: {output}")
   
    print(f"The model output is: {output:.2f}")

