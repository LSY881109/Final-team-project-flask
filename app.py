from flask import Flask, request, jsonify, render_template
import os
import time
import json  # JSON 응답을 더 깔끔하게 만들기 위해 import

# from werkzeug.utils import secure_filename # 실제 파일 이름 보안 처리 시 필요

app = Flask(__name__)
# Flask 서버를 실행할 포트 (Spring Boot의 application.properties와 일치해야 함)
FLASK_PORT = 5000


# Spring Boot의 application.properties에 설정된 flask.api.url=http://localhost:5000 에 대응

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

    # 2. 파일 정보 확인 (실제 AI 모델은 여기서 이미지 파일을 로드하고 분석합니다)
    filename = image_file.filename
    print(f"Received file: {filename}")

    # 3. AI 분석 흉내 (더미 로직)
    # 실제 분석 시간 지연을 흉내냅니다.
    time.sleep(1)

    # 파일 이름에 따라 더미 결과를 다르게 반환하는 로직 (테스트 용)
    if 'chicken' in filename.lower():
        food_name = "후라이드 치킨"
        calories = 650
        confidence = 0.95
        recipe_id = "mongodb_id_chicken_001"
        youtube_link = "https://www.youtube.com/watch?v=chicken_recipe"
    elif 'salad' in filename.lower():
        food_name = "닭가슴살 샐러드"
        calories = 300
        confidence = 0.88
        recipe_id = "mongodb_id_salad_002"
        youtube_link = "https://www.youtube.com/watch?v=salad_recipe"
    else:
        food_name = "인식할 수 없는 음식"
        calories = 0
        confidence = 0.55
        recipe_id = "N/A"
        youtube_link = "N/A"

    # 4. Spring Boot의 FoodAnalysisResultDTO 형식에 맞춘 JSON 응답 구성
    response_data = {
        "recognizedFoodName": food_name,
        "confidenceScore": confidence,
        "servingSizeInfo": "Based on image analysis (approx. 1 serving)",
        "totalCalories": calories,
        "macroNutrientsSummary": "단백질: 30g, 지방: 25g, 탄수화물: 40g",  # 예시 데이터
        "recipeId": recipe_id,
        "externalRecipeLink": youtube_link,
        "message": f"AI analysis complete for {food_name}"
    }

    # 5. JSON 응답 반환
    return jsonify(response_data), 200


# ----------------------------------------------------
# 🚩 서버 실행
# ----------------------------------------------------
if __name__ == '__main__':
    print(f"Flask server is running on http://127.0.0.1:{FLASK_PORT}")
    # debug=True는 개발용입니다. (배포 시에는 끄세요)
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)