# 백엔드 테스트 가이드

## 📋 테스트 전 준비사항

### 1. Flask 서버 실행
```bash
cd Final-team-project-flask
python app.py
```

서버가 정상적으로 실행되면:
```
🍎 Apple Silicon GPU (MPS) 사용!
🤖 BiRefNet 배경 제거 모델 로딩 중...
✅ BiRefNet 배경 제거 모델 로딩 완료!
Flask server is running on http://127.0.0.1:5001
```

### 2. Spring Boot 서버 실행
```bash
cd Final-team-project-back
./gradlew bootRun
# 또는
./gradlew build
java -jar build/libs/2_Team_back-0.0.1-SNAPSHOT.jar
```

## 🔗 엔드포인트 확인

### Flask 서버
- **테스트 페이지**: http://localhost:5001
- **분류 API (테스트용)**: POST http://localhost:5001/classify
- **분석 API (Spring Boot용)**: POST http://localhost:5001/analyze

### Spring Boot 서버
- **분석 API**: POST http://localhost:8080/api/analysis
  - 요청 파라미터:
    - `userId`: Long (사용자 ID)
    - `image`: MultipartFile (이미지 파일)

## 🧪 테스트 방법

### 방법 1: Flask 테스트 페이지 (가장 간단)
1. 브라우저에서 http://localhost:5001 접속
2. 이미지 업로드
3. "이미지 분석하기" 클릭
4. 결과 확인

### 방법 2: curl로 Flask 직접 테스트
```bash
curl -X POST http://localhost:5001/analyze \
  -F "image=@/path/to/your/image.jpg"
```

예상 응답:
```json
{
  "class": "파스타",
  "confidence": 95.5
}
```

### 방법 3: curl로 Spring Boot 통합 테스트
```bash
curl -X POST http://localhost:8080/api/analysis \
  -F "userId=1" \
  -F "image=@/path/to/your/image.jpg"
```

예상 응답:
```json
{
  "foodName": "파스타",
  "accuracy": 0.955,
  "nutritionData": { ... },
  "youtubeRecipes": [ ... ],
  "message": "AI analysis complete for 파스타"
}
```

### 방법 4: Postman 사용
1. **Flask 테스트**:
   - Method: POST
   - URL: http://localhost:5001/analyze
   - Body: form-data
     - Key: `image` (type: File)
     - Value: 이미지 파일 선택

2. **Spring Boot 테스트**:
   - Method: POST
   - URL: http://localhost:8080/api/analysis
   - Body: form-data
     - Key: `userId` (type: Text), Value: `1`
     - Key: `image` (type: File), Value: 이미지 파일 선택

## ⚠️ 문제 해결

### Flask 서버가 시작되지 않는 경우
- 포트 5001이 이미 사용 중인지 확인: `lsof -ti:5001`
- 필요한 패키지 설치 확인: `pip install -r requirements.txt`

### Spring Boot에서 Flask 연결 실패
- `application.properties`에서 Flask URL 확인: `flask.api.url=http://localhost:5001/analyze`
- Flask 서버가 실행 중인지 확인
- 네트워크 방화벽 확인

### 모델 로딩 실패
- EfficientNet 모델 파일 확인: `efficientnet_finetuned_best.pth`
- BiRefNet 모델이 자동 다운로드되는지 확인 (첫 실행 시 시간 소요)

## 📝 체크리스트

- [ ] Flask 서버 실행 확인 (포트 5001)
- [ ] Spring Boot 서버 실행 확인 (포트 8080)
- [ ] `application.properties`의 Flask URL 확인
- [ ] 테스트 이미지 준비
- [ ] Flask 테스트 페이지에서 정상 동작 확인
- [ ] Spring Boot API 통합 테스트 확인

