# Công dụng: Train model phân loại topic với TF-IDF + Naive Bayes

import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from underthesea import word_tokenize
import re

def clean_text(text):
    """Làm sạch văn bản"""
    # Chuyển về chữ thường
    text = text.lower()
    # Loại bỏ ký tự đặc biệt, giữ lại chữ cái tiếng Việt
    text = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    return text

def tokenize_text(text):
    """Tách từ tiếng Việt"""
    try:
        tokens = word_tokenize(text, format="text")
        return tokens
    except:
        return text

def load_dataset(file_path):
    """Đọc dataset"""
    print(f"📂 Đang đọc dataset từ: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Đã đọc {len(df)} mẫu")
    print(f"\n📊 Phân bố topics:")
    print(df['topic'].value_counts())
    return df

def preprocess_data(df):
    """Tiền xử lý dữ liệu"""
    print("\n🧹 Đang tiền xử lý dữ liệu...")
    
    # Làm sạch văn bản
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Tách từ tiếng Việt
    print("✂️  Đang tách từ tiếng Việt (có thể mất 1-2 phút)...")
    df['text_tokenized'] = df['text_clean'].apply(tokenize_text)
    
    print("✅ Hoàn thành tiền xử lý")
    return df

def train_model(df):
    """Train model phân loại"""
    print("\n🤖 BẮT ĐẦU TRAIN MODEL.. .\n")
    
    # Chia train/test
    X = df['text_tokenized']
    y = df['topic']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Dữ liệu train: {len(X_train)} mẫu")
    print(f"📊 Dữ liệu test: {len(X_test)} mẫu")
    
    # TF-IDF Vectorizer
    print("\n🔤 Đang tạo TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✅ TF-IDF shape: {X_train_tfidf.shape}")
    
    # Train Naive Bayes
    print("\n🎓 Đang train Multinomial Naive Bayes...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)
    
    print("✅ Train model hoàn thành!")
    
    # Đánh giá
    print("\n📈 ĐÁNH GIÁ MODEL:\n")
    
    # Accuracy
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\n📊 Chi tiết theo từng topic:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion Matrix
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return model, vectorizer, accuracy

def save_model(model, vectorizer, output_dir='models'):
    """Lưu model và vectorizer"""
    print(f"\n💾 Đang lưu model...")
    
    # Tạo thư mục nếu chưa có
    Path(output_dir).mkdir(exist_ok=True)
    
    # Lưu model
    model_path = Path(output_dir) / 'topic_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model đã lưu tại: {model_path}")
    
    # Lưu vectorizer
    vectorizer_path = Path(output_dir) / 'vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✅ Vectorizer đã lưu tại: {vectorizer_path}")

def test_prediction(model, vectorizer):
    """Test thử dự đoán"""
    print("\n🧪 TEST THỬ DỰ ĐOÁN:\n")
    
    test_texts = [
        "Đội tuyển Việt Nam giành chiến thắng 2-0 trong trận đấu vòng loại World Cup hôm qua",
        "Giá vàng hôm nay tăng mạnh do ảnh hưởng của thị trường thế giới",
        "Apple vừa ra mắt iPhone mới với nhiều tính năng công nghệ đột phá"
    ]
    
    for text in test_texts:
        # Tiền xử lý
        text_clean = clean_text(text)
        text_tokenized = tokenize_text(text_clean)
        
        # Vectorize
        text_tfidf = vectorizer. transform([text_tokenized])
        
        # Dự đoán
        prediction = model.predict(text_tfidf)[0]
        proba = model.predict_proba(text_tfidf)[0]
        
        # Lấy top 3 topics
        top_indices = proba.argsort()[-3:][::-1]
        top_topics = [(model.classes_[i], proba[i]*100) for i in top_indices]
        
        print(f"📝 Text: {text[:70]}...")
        print(f"🎯 Dự đoán: {prediction}")
        print(f"📊 Top 3 topics:")
        for topic, prob in top_topics:
            print(f"   - {topic}: {prob:.2f}%")
        print()

def main():
    print("="*70)
    print("TRAIN MODEL PHÂN LOẠI TOPIC VĂN BẢN TIẾNG VIỆT")
    print("="*70)
    
    # Load dataset
    dataset_path = 'data/processed_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Không tìm thấy file: {dataset_path}")
        print("💡 Chạy download_dataset.py trước!")
        return
    
    df = load_dataset(dataset_path)
    
    # Preprocess
    df = preprocess_data(df)
    
    # Train
    model, vectorizer, accuracy = train_model(df)
    
    # Save
    save_model(model, vectorizer)
    
    # Test
    test_prediction(model, vectorizer)
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH BƯỚC 2.2 - TRAIN MODEL!")
    print("="*70)
    print(f"\n📊 Kết quả:")
    print(f"   - Accuracy: {accuracy*100:.2f}%")
    print(f"   - Model: models/topic_classifier.pkl")
    print(f"   - Vectorizer: models/vectorizer. pkl")
    print(f"\n➡️  Tiếp theo: Tạo API backend (app.py)")

if __name__ == "__main__":
    main()