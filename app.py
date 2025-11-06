from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import time
import json  # JSON 응답을 더 깔끔하게 만들기 위해 import
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import gc

# from werkzeug.utils import secure_filename # 실제 파일 이름 보안 처리 시 필요

app = Flask(__name__)
# CORS 설정: Spring Boot 서버와 React 개발 서버에서의 요청 허용
# 모든 엔드포인트에 대해 CORS 허용 (개발용)
CORS(app, origins=[
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173"
], methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])
# Flask 서버를 실행할 포트 (Spring Boot의 application.properties와 일치해야 함)
# macOS의 AirPlay Receiver가 5000 포트를 사용하므로 5001로 변경
FLASK_PORT = 5001

# 클래스 이름
class_names = ['감바스', '숯불치킨', '양념치킨', '파스타', '후라이드치킨']

# 디바이스 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Apple Silicon GPU (MPS) 사용!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🔥 CUDA GPU 사용!")
else:
    device = torch.device("cpu")
    print("💻 CPU 사용")

# EfficientNet 모델 로드 함수
def load_efficientnet(model_path="efficientnet_finetuned_best.pth", num_classes=5):
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# BiRefNet 배경 제거 모델 로드 (기본값: 활성화)
ENABLE_BACKGROUND_REMOVAL = True  # False로 변경하면 배경 제거 비활성화

birefnet_model = None
if ENABLE_BACKGROUND_REMOVAL:
    try:
        print("🤖 BiRefNet 배경 제거 모델 로딩 중...")
        from transformers import AutoModelForImageSegmentation
        
        # 메모리 정리
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet',
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type != 'cpu' else torch.float32,
            low_cpu_mem_usage=True
        )
        birefnet_model.to(device)
        birefnet_model.eval()
        
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print("✅ BiRefNet 배경 제거 모델 로딩 완료!")
    except ImportError as e:
        print(f"⚠️  transformers 모듈이 설치되지 않았습니다: {e}")
        print("⚠️  배경 제거 없이 진행합니다.")
        print("💡 pip install transformers 로 설치하세요.")
        birefnet_model = None
    except Exception as e:
        print(f"⚠️  BiRefNet 로딩 실패: {e}")
        print("⚠️  배경 제거 없이 진행합니다.")
        birefnet_model = None
else:
    print("ℹ️  배경 제거 기능 비활성화됨 (빠른 시작)")

# 모델 로드
model = load_efficientnet()

# 배경 제거 함수 (BiRefNet 사용)
def remove_background(image, birefnet_model, device):
    """
    BiRefNet을 사용하여 배경 제거
    """
    if birefnet_model is None:
        return image
    
    try:
        # BiRefNet 입력 전처리
        input_size = (1024, 1024)
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        # float16으로 변환 (메모리 절약)
        if device.type != 'cpu':
            input_tensor = input_tensor.half()
        
        # 배경 제거
        with torch.no_grad():
            preds = birefnet_model(input_tensor)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        
        # RGBA로 변환 (알파 채널에 마스크 적용)
        image_rgba = image.convert("RGBA")
        image_rgba.putalpha(mask)
        
        # 흰색 배경으로 합성
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        white_bg.paste(image_rgba, (0, 0), image_rgba)
        
        return white_bg
    except Exception as e:
        print(f"배경 제거 중 오류 발생: {e}")
        return image

# 이미지 전처리 함수 (배경 제거 + 리사이즈 + 패딩)
def preprocess_image(image, remove_bg=True):
    """
    이미지 전처리: 배경 제거 → 리사이즈 → 패딩 → 정규화
    remove_bg: 배경 제거 사용 여부 (기본값: True - BiRefNet 사용)
    """
    original_size = image.size
    
    # 1. 배경 제거 (기본값: 활성화 - BiRefNet 사용)
    if remove_bg and birefnet_model is not None:
        image = remove_background(image, birefnet_model, device)
    
    # 2. 224x224로 리사이즈 + 패딩 (비율 유지)
    target_size = (224, 224)
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # 패딩 추가 (흰색 배경)
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    paste_position = ((target_size[0] - image.size[0]) // 2,
                      (target_size[1] - image.size[1]) // 2)
    new_image.paste(image, paste_position)
    
    # 3. 모델 입력용 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(new_image).unsqueeze(0).to(device)


# Spring Boot의 application.properties에 설정된 flask.api.url=http://localhost:5000 에 대응

# ----------------------------------------------------
# 🚩 테스트용 웹페이지 렌더링
# ----------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

# ----------------------------------------------------
# 🚩 테스트용 이미지 분류 API
# ----------------------------------------------------
@app.route('/classify', methods=['POST'])
def classify_image():
    """테스트용 이미지 분류 API"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400

    image_file = request.files['image']
    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # 이미지 전처리 및 모델 예측 (배경 제거 기본값: 활성화)
        image = Image.open(image_file).convert('RGB')
        # 배경 제거 옵션 (기본값: true, false로 비활성화 가능)
        remove_bg = request.args.get('remove_bg', 'true').lower() == 'true'
        image_tensor = preprocess_image(image, remove_bg=remove_bg)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()

        # 상위 3개 결과
        top3_probs, top3_indices = torch.topk(probabilities, min(3, len(class_names)))
        
        results = []
        for i in range(len(top3_indices)):
            results.append({
                "class": class_names[top3_indices[i].item()],
                "confidence": round(top3_probs[i].item() * 100, 2)
            })

        return jsonify({
            "predicted_class": class_names[predicted_idx],
            "confidence": round(confidence * 100, 2),
            "top3": results
        }), 200
        
    except Exception as e:
        print(f"Error during image classification: {str(e)}")
        return jsonify({"error": f"Error during image classification: {str(e)}"}), 500

# ----------------------------------------------------
# 🚩 음식 분석 API 엔드포인트: Spring Boot의 AIAnalysisService가 호출할 경로
# ----------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze_image():
    # 1. Spring Boot로부터 요청 수신 확인
    if 'image' not in request.files:
        print("Error: 'image' file part not found in the request.")
        return jsonify({"message": "No image file provided"}), 400

    image_file = request.files['image']
    
    if image_file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    # 2. 파일 정보 확인
    filename = image_file.filename
    print(f"Received file: {filename}")

    try:
        # 3. 이미지 전처리 및 모델 예측 (배경 제거는 기본적으로 비활성화 - 빠른 처리)
        image = Image.open(image_file).convert('RGB')
        # 배경 제거는 느리므로 기본값 False, 필요시 쿼리 파라미터로 활성화 가능
        remove_bg = request.args.get('remove_bg', 'false').lower() == 'true'
        image_tensor = preprocess_image(image, remove_bg=remove_bg)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()

        # 4. 예측 결과 (Spring Boot의 AiResponse DTO에 맞춰 두 가지만 반환)
        food_name = class_names[predicted_idx]
        confidence_score = round(confidence * 100, 2)  # 0.0~1.0 → 0.0~100.0으로 변환

        # 5. Spring Boot의 AiResponse DTO 형식에 맞춘 JSON 응답 구성
        # Spring Boot에서 나머지 정보(칼로리, 영양정보, 레시피 등)는 DB에서 가져와서 처리
        response_data = {
            "class": food_name,  # AiResponse의 predictedClass 필드와 매핑
            "confidence": confidence_score  # AiResponse의 confidence 필드와 매핑
        }

        # 6. JSON 응답 반환
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error during image analysis: {str(e)}")
        return jsonify({
            "class": "인식할 수 없는 음식",
            "confidence": 0.0
        }), 500


# ----------------------------------------------------
# 🚩 서버 실행
# ----------------------------------------------------
if __name__ == '__main__':
    print(f"Flask server is running on http://127.0.0.1:{FLASK_PORT}")
    # debug=True는 개발용입니다. (배포 시에는 끄세요)
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)