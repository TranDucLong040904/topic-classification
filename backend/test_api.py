# Công dụng: Test API hoạt động đúng không

import requests
import json

# URL của API
API_URL = "http://localhost:5000"

def test_home():
    """Test endpoint /"""
    print("="*60)
    print("TEST 1: Kiểm tra API hoạt động")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print("✅ PASS\n")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return False

def test_health():
    """Test endpoint /health"""
    print("="*60)
    print("TEST 2: Kiểm tra sức khỏe API")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status code: {response. status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print("✅ PASS\n")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return False

def test_predict(text, expected_topic=None):
    """Test endpoint /predict"""
    print("="*60)
    print(f"TEST: Dự đoán topic")
    print("="*60)
    print(f"Input: {text[:80]}...")
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Dự đoán thành công!")
            print(f"Top topic: {result['top_topic']}")
            print(f"\nTop 5 predictions:")
            for pred in result['predictions']:
                print(f"   - {pred['topic']}: {pred['probability']}%")
            
            if expected_topic and result['top_topic'] == expected_topic:
                print(f"\n✅ PASS: Dự đoán đúng {expected_topic}")
            elif expected_topic:
                print(f"\n⚠️  WARN: Dự đoán {result['top_topic']}, mong đợi {expected_topic}")
            else:
                print(f"\n✅ PASS")
            
            print()
            return True
        else:
            print(f"❌ FAIL: {response.json()}\n")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return False

def test_predict_empty():
    """Test với văn bản rỗng"""
    print("="*60)
    print("TEST: Văn bản rỗng (phải lỗi)")
    print("="*60)
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": ""},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            print(f"✅ PASS: API trả về lỗi đúng")
            print(f"Message: {response.json()['message']}\n")
            return True
        else:
            print(f"❌ FAIL: Không trả về lỗi\n")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {e}\n")
        return False

def main():
    print("\n" + "="*60)
    print("🧪 TEST API TOPIC CLASSIFICATION")
    print("="*60)
    print("\n⚠️  Đảm bảo API đang chạy tại http://localhost:5000\n")
    
    results = []
    
    # Test 1: Home
    results.append(test_home())
    
    # Test 2: Health
    results.append(test_health())
    
    # Test 3: Văn bản rỗng
    results.append(test_predict_empty())
    
    # Test 4: Thể thao
    results.append(test_predict(
        "Đội tuyển Việt Nam giành chiến thắng 3-0 trước Thái Lan trong trận đấu vòng loại World Cup.  Các cầu thủ đã thể hiện phong độ tuyệt vời.",
        expected_topic="Thể thao"
    ))
    
    # Test 5: Công nghệ
    results.append(test_predict(
        "Apple vừa ra mắt iPhone thế hệ mới với chip xử lý mạnh mẽ và camera chất lượng cao. Sản phẩm được trang bị công nghệ AI tiên tiến.",
        expected_topic="Công nghệ"
    ))
    
    # Test 6: Kinh tế
    results.append(test_predict(
        "Giá vàng trong nước tăng cao theo đà tăng của giá vàng thế giới. Nhiều người đổ xô đi mua vàng để đầu tư và tích trữ.",
        expected_topic="Kinh tế"
    ))
    
    # Test 7: Sức khỏe
    results.append(test_predict(
        "Bộ Y tế khuyến cáo người dân cần tiêm vaccine phòng ngừa dịch bệnh.  Vaccine đã được chứng minh là an toàn và hiệu quả cao.",
        expected_topic="Sức khỏe"
    ))
    
    # Tổng kết
    print("="*60)
    print("📊 TỔNG KẾT")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 TẤT CẢ TEST ĐỀU PASS!")
    else:
        print("\n⚠️  MỘT SỐ TEST BỊ FAIL")
    
    print("="*60)

if __name__ == "__main__":
    main()