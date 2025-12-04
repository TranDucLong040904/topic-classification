import os
import pandas as pd
import re
from pathlib import Path

# Mapping VNTC topics sang 15 topics của chúng ta
TOPIC_MAPPING = {
    'Chinh tri Xa hoi': 'Chính trị',
    'Doi song': 'Thời sự',
    'Khoa hoc': 'Khoa học',
    'Kinh doanh': 'Kinh tế',
    'Phap luat': 'Pháp luật',
    'Suc khoe': 'Sức khỏe',
    'The gioi': 'Thời sự',
    'The thao': 'Thể thao',
    'Van hoa': 'Văn hóa',
    'Vi tinh': 'Công nghệ',
}

def count_words(text):
    """Đếm số từ trong văn bản tiếng Việt"""
    words = text.split()
    return len(words)

def clean_text(text):
    """Làm sạch văn bản"""
    text = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', ' ', text)
    text = ' '.join(text.split())
    return text. strip()

def read_vntc_data(vntc_path):
    """Đọc dữ liệu từ VNTC dataset"""
    data = []
    
    # Thử tìm trong Ver1.1/Train/
    train_path = Path(vntc_path) / 'Data' / '10Topics' / 'Ver1.1' / 'Train'
    
    if not train_path.exists():
        print(f"❌ Không tìm thấy thư mục: {train_path}")
        return pd.DataFrame()
    
    print(f"📂 Đang đọc từ: {train_path}\n")
    
    # Duyệt qua các thư mục topic
    for topic_dir in train_path.iterdir():
        if not topic_dir.is_dir():
            continue
            
        topic_name = topic_dir.name
        print(f"📂 Đang đọc topic: {topic_name}")
        count = 0
        
        # Đọc các file văn bản
        for file_path in topic_dir.glob('*.txt'):
            try:
                # Thử nhiều encoding
                text = None
                for encoding in ['utf-16', 'utf-8', 'utf-16-le', 'utf-16-be', 'latin-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                            text = f.read()
                            if text and len(text.strip()) > 10:  # Đảm bảo có nội dung
                                break
                    except:
                        continue
                
                if text and len(text.strip()) > 10:
                    data.append({
                        'text': text.strip(),
                        'original_topic': topic_name
                    })
                    count += 1
                    
            except Exception as e:
                continue
        
        print(f"   ✅ Đọc được {count} văn bản")
    
    print(f"\n✅ Tổng cộng đã đọc {len(data)} văn bản")
    return pd.DataFrame(data)

def process_dataset(vntc_path, output_path, min_words=50, max_words=300, samples_per_topic=50):
    """Xử lý dataset và tạo file CSV chuẩn"""
    
    print("🔄 Bắt đầu xử lý dataset.. .\n")
    
    # Đọc dữ liệu VNTC
    df = read_vntc_data(vntc_path)
    
    if df.empty:
        print("❌ Không có dữ liệu để xử lý")
        print("\n💡 Kiểm tra lại cấu trúc thư mục:")
        print("   backend/data/VNTC/Data/10Topics/Ver1.1/Train/")
        return None
    
    # Làm sạch văn bản
    print("\n🧹 Đang làm sạch văn bản...")
    df['text'] = df['text'].apply(clean_text)
    
    # Loại bỏ văn bản trống
    df = df[df['text'].str.len() > 0]
    
    # Đếm số từ
    df['word_count'] = df['text'].apply(count_words)
    
    print(f"📊 Phân bố độ dài văn bản:")
    print(f"   Min: {df['word_count']. min()} từ")
    print(f"   Max: {df['word_count'].max()} từ")
    print(f"   Trung bình: {df['word_count'].mean():.0f} từ")
    
    # Lọc theo độ dài
    df_filtered = df[(df['word_count'] >= min_words) & (df['word_count'] <= max_words)]
    print(f"\n✅ Còn {len(df_filtered)} văn bản sau khi lọc độ dài ({min_words}-{max_words} từ)")
    
    if len(df_filtered) == 0:
        print("❌ Không còn dữ liệu sau khi lọc!")
        print(f"💡 Thử giảm min_words hoặc tăng max_words")
        return None
    
    # Mapping topics
    df_filtered['topic'] = df_filtered['original_topic']. map(TOPIC_MAPPING)
    
    # Loại bỏ các topic không map được
    df_filtered = df_filtered[df_filtered['topic']. notna()]
    
    print(f"\n📊 Phân bố trước khi cân bằng:")
    print(df_filtered['topic'].value_counts())
    
    # Cân bằng dữ liệu
    balanced_data = []
    print(f"\n⚖️  Đang cân bằng dữ liệu (tối đa {samples_per_topic} mẫu/topic):")
    for topic in df_filtered['topic'].unique():
        topic_data = df_filtered[df_filtered['topic'] == topic]
        n_samples = min(len(topic_data), samples_per_topic)
        
        if len(topic_data) >= samples_per_topic:
            topic_data = topic_data.sample(n=samples_per_topic, random_state=42)
        else:
            topic_data = topic_data.sample(frac=1, random_state=42)
            
        balanced_data.append(topic_data)
        print(f"   {topic}: {n_samples} mẫu")
    
    df_balanced = pd.concat(balanced_data, ignore_index=True)
    
    # Chỉ giữ lại cột cần thiết
    df_final = df_balanced[['text', 'topic']]
    
    # Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Lưu file
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Đã lưu {len(df_final)} mẫu vào: {output_path}")
    
    # Thống kê cuối cùng
    print(f"\n📊 THỐNG KÊ DATASET CUỐI CÙNG:")
    print(f"Tổng số mẫu: {len(df_final)}")
    print(f"\nPhân bố theo topic:")
    print(df_final['topic'].value_counts())
    
    # Hiển thị 2 mẫu ngẫu nhiên
    print(f"\n📝 MẪU DỮ LIỆU:")
    for i, row in df_final. sample(2). iterrows():
        print(f"\n--- Mẫu {i+1} ---")
        print(f"Topic: {row['topic']}")
        print(f"Text: {row['text'][:150]}...")
    
    return df_final

if __name__ == "__main__":
    # Đường dẫn
    VNTC_PATH = "data/VNTC"
    OUTPUT_PATH = "data/processed_dataset.csv"
    
    # Xử lý
    df = process_dataset(
        vntc_path=VNTC_PATH,
        output_path=OUTPUT_PATH,
        min_words=50,
        max_words=300,
        samples_per_topic=50
    )
    
    if df is not None and not df.empty:
        print("\n" + "="*60)
        print("✅ HOÀN THÀNH BƯỚC 1. 1 & 1.2!")
        print("="*60)
        print(f"\n💾 File đã lưu tại: {OUTPUT_PATH}")
        print(f"📊 Tổng số mẫu: {len(df)}")
        print(f"\n➡️  Tiếp theo: Train model (Bước 2. 2)")
    else:
        print("\n❌ Xử lý thất bại!")