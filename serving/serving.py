import json
import numpy as np
import tensorflow as tf
from PIL import Image
import io

def input_handler(data, context):
    """Chuyển đổi dữ liệu ảnh binary từ người dùng gửi lên thành mảng Numpy"""
    if context.request_content_type == 'application/x-image':
        image = Image.open(io.BytesIO(data.read())).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        return json.dumps({"instances": img_array.tolist()})
    raise ValueError("Chỉ chấp nhận application/x-image")

def output_handler(data, context):
    """Lấy kết quả từ model và trả về chữ Cat/Dog cho dễ đọc"""
    response_body = data.json()
    predictions = response_body['predictions']
    score = predictions[0][0]
    label = "Dog" if score > 0.5 else "Cat"
    return json.dumps({"label": label, "score": float(score), "raw_predictions": predictions})