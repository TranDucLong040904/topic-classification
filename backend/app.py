# Công dụng: Flask API server, nhận văn bản từ frontend, trả về TOP 5 topics với xác suất

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import os
from underthesea import word_tokenize
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Cho phép frontend gọi API

# Load model và vectorizer khi khởi động
MODEL_PATH = Path('models/topic_classifier.pkl')
VECTORIZER_PATH = Path('models/vectorizer.pkl')

print("🔄 Đang load model...")

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Load model thành công!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    model = None
    vectorizer = None

def clean_text(text):
    """Làm sạch văn bản"""
    text = text.lower()
    text = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
    text = ' '.join(text.split())
    return text

def tokenize_text(text):
    """Tách từ tiếng Việt"""
    try:
        tokens = word_tokenize(text, format="text")
        return tokens
    except:
        return text

@app.route('/')
def home():
    """Endpoint kiểm tra API hoạt động"""
    return jsonify({
        'status': 'success',
        'message': 'Topic Classification API is running! ',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    API dự đoán topic
    
    Input JSON:
    {
        "text": "Văn bản cần phân loại"
    }
    
    Output JSON:
    {
        "status": "success",
        "predictions": [
            {"topic": "Thể thao", "probability": 76.55},
            {"topic": "Kinh tế", "probability": 12.30},
            ... 
        ],
        "top_topic": "Thể thao"
    }
    """
    
    # Kiểm tra model đã load chưa
    if model is None or vectorizer is None:
        return jsonify({
            'status': 'error',
            'message': 'Model chưa được load.  Vui lòng train model trước!'
        }), 500
    
    # Lấy dữ liệu từ request
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        # Validate input
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Vui lòng nhập văn bản!'
            }), 400
        
        if len(text) < 10:
            return jsonify({
                'status': 'error',
                'message': 'Văn bản quá ngắn!  Vui lòng nhập ít nhất 10 ký tự.'
            }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Lỗi đọc dữ liệu: {str(e)}'
        }), 400
    
    # Tiền xử lý văn bản
    try:
        text_clean = clean_text(text)
        text_tokenized = tokenize_text(text_clean)
        
        # Vectorize
        text_tfidf = vectorizer.transform([text_tokenized])
        
        # Dự đoán
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        
        # Lấy top 5 topics
        top_indices = probabilities.argsort()[-5:][::-1]
        predictions = []
        
        for idx in top_indices:
            predictions.append({
                'topic': model.classes_[idx],
                'probability': round(probabilities[idx] * 100, 2)
            })
        
        # Trả về kết quả
        return jsonify({
            'status': 'success',
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'predictions': predictions,
            'top_topic': prediction
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Lỗi xử lý: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint kiểm tra sức khỏe của API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 TOPIC CLASSIFICATION API")
    print("="*60)
    print(f"📍 URL: http://localhost:5000")
    print(f"📍 Endpoints:")
    print(f"   - GET  /          : Kiểm tra API")
    print(f"   - POST /predict   : Dự đoán topic")
    print(f"   - GET  /health    : Kiểm tra sức khỏe")
    print("="*60 + "\n")
    
    # Use Render-provided PORT when deployed; default 5000 for local dev (Song ngu: Dung PORT tu Render, fallback 5000 khi chay local)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)