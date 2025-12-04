# Công dụng: Tải dataset Vietnamese News từ Kaggle hoặc tạo dữ liệu mẫu
 
import pandas as pd
import requests
from pathlib import Path

def create_sample_dataset():
    """Tạo dataset mẫu cho 10 topics có trong VNTC"""
    
    print("📝 Đang tạo dataset mẫu...")
    
    # Dữ liệu mẫu cho mỗi topic
    sample_data = {
        'Thể thao': [
            'Đội tuyển Việt Nam giành chiến thắng 3-0 trước Thái Lan trong trận đấu vòng loại World Cup.  Các cầu thủ đã thể hiện phong độ tuyệt vời và tinh thần chiến đấu cao.  Huấn luyện viên Park Hang-seo rất hài lòng với kết quả này.  Trận đấu diễn ra trên sân Mỹ Đình với sự cổ vũ nhiệt tình của hàng vạn cổ động viên.',
            'Giải bóng đá ngoại hạng Anh hấp dẫn với nhiều trận cầu đỉnh cao. Manchester City đang dẫn đầu bảng xếp hạng với lối chơi ấn tượng. Liverpool và Arsenal cũng đang có phong độ tốt.  Cuộc đua vô địch năm nay hứa hẹn sẽ rất căng thẳng đến những vòng đấu cuối cùng của mùa giải.',
            'Quần vợt Việt Nam có bước tiến mới với nhiều tay vợt trẻ tài năng. Lý Hoàng Nam tiếp tục thi đấu ấn tượng tại các giải quốc tế. Liên đoàn quần vợt đang có kế hoạch đầu tư phát triển bộ môn này. Hy vọng Việt Nam sẽ có thêm nhiều vận động viên xuất sắc trong tương lai.',
            'SEA Games năm nay Việt Nam đặt mục tiêu vào top 3 tổng sắp.  Các vận động viên đang tập luyện chăm chỉ để chuẩn bị cho giải đấu. Đoàn thể thao Việt Nam được đầu tư tốt về cơ sở vật chất và huấn luyện viên.  Người hâm mộ đang rất kỳ vọng vào thành tích của đoàn.',
            'Giải marathon quốc tế TP. HCM thu hút hàng nghìn vận động viên tham gia. Đây là sự kiện thể thao lớn nhất trong năm tại thành phố. Các vận động viên đến từ nhiều quốc gia trên thế giới.  Không khí sôi động và tinh thần thể thao được lan tỏa mạnh mẽ.',
        ],
        'Kinh tế': [
            'Nền kinh tế Việt Nam tăng trưởng ấn tượng trong quý đầu năm đạt 6. 5 phần trăm. Xuất khẩu và đầu tư nước ngoài đều có sự tăng trưởng tích cực. Chính phủ đang triển khai nhiều chính sách hỗ trợ doanh nghiệp.  Các chuyên gia dự báo kinh tế sẽ tiếp tục phát triển trong thời gian tới.',
            'Thị trường chứng khoán Việt Nam biến động mạnh trong tuần qua. VN-Index giảm xuống mức thấp nhất trong hai tháng. Nhiều cổ phiếu ngân hàng và bất động sản bị bán tháo. Các nhà đầu tư đang thận trọng chờ đợi tín hiệu tích cực từ thị trường.',
            'Giá vàng trong nước tăng cao theo đà tăng của giá vàng thế giới. Nhiều người đổ xô đi mua vàng để đầu tư và tích trữ. Các chuyên gia khuyên nhà đầu tư nên thận trọng vì thị trường biến động phức tạp. Ngân hàng Nhà nước đang theo dõi sát diễn biến giá vàng.',
            'Ngân hàng trung ương điều chỉnh lãi suất để kiểm soát lạm phát. Quyết định này ảnh hưởng đến hoạt động cho vay của các ngân hàng thương mại. Doanh nghiệp và người dân cần cân nhắc kỹ kế hoạch vay vốn. Chuyên gia cho rằng đây là động thái cần thiết trong giai đoạn hiện tại.',
            'Đầu tư nước ngoài vào Việt Nam tăng mạnh trong năm nay. Nhiều dự án lớn được khởi công trong các khu công nghiệp và khu công nghệ cao. Chính phủ cam kết tạo môi trường đầu tư thuận lợi cho nhà đầu tư.  Việt Nam đang là điểm đến hấp dẫn cho dòng vốn nước ngoài.',
        ],
        'Công nghệ': [
            'Trí tuệ nhân tạo đang thay đổi nhiều ngành công nghiệp trên toàn thế giới.  Các ứng dụng AI ngày càng phổ biến trong đời sống hàng ngày. Việt Nam cũng bắt đầu ứng dụng AI vào nhiều lĩnh vực khác nhau. Các chuyên gia dự báo AI sẽ tạo ra cuộc cách mạng công nghệ mới.',
            'Apple vừa ra mắt iPhone thế hệ mới với nhiều tính năng đột phá. Sản phẩm được trang bị chip xử lý mạnh mẽ và camera chất lượng cao. Người dùng Việt Nam rất quan tâm và mong chờ sản phẩm này.  Giá bán dự kiến sẽ cao hơn thế hệ trước khoảng 10 phần trăm.',
            'Mạng 5G đang được triển khai rộng rãi tại các thành phố lớn ở Việt Nam. Tốc độ internet tăng gấp nhiều lần so với 4G mang lại trải nghiệm tốt hơn. Các nhà mạng đang cạnh tranh để thu hút khách hàng sử dụng dịch vụ. Dự kiến đến cuối năm sẽ có hàng triệu thuê bao 5G.',
            'Xe điện đang trở thành xu hướng mới trong ngành công nghiệp ô tô. VinFast đã xuất khẩu hàng nghìn xe điện sang thị trường Mỹ và châu Âu. Công nghệ pin và trạm sạc đang được đầu tư phát triển mạnh. Xe điện được kỳ vọng sẽ thay thế dần xe xăng trong tương lai gần.',
            'Công nghệ blockchain đang được ứng dụng trong nhiều lĩnh vực khác nhau. Ngân hàng sử dụng blockchain để tăng tính bảo mật trong giao dịch. Các startup công nghệ cũng tích cực nghiên cứu và phát triển ứng dụng blockchain.  Đây được coi là công nghệ của tương lai với tiềm năng rất lớn.',
        ],
        'Sức khỏe': [
            'Bộ Y tế khuyến cáo người dân cần tiêm vaccine phòng ngừa dịch bệnh. Vaccine đã được chứng minh là an toàn và hiệu quả cao. Các trung tâm y tế đang tổ chức tiêm chủng rộng rãi cho cộng đồng. Việc tiêm vaccine giúp bảo vệ sức khỏe bản thân và người thân.',
            'Chế độ ăn uống lành mạnh rất quan trọng để duy trì sức khỏe tốt. Nên ăn nhiều rau xanh và trái cây tươi mỗi ngày. Hạn chế thức ăn nhiều dầu mỡ và đường để tránh béo phì.  Uống đủ nước và tập thể dục đều đặn cũng rất cần thiết.',
            'Bệnh tiểu đường đang gia tăng ở Việt Nam do lối sống không lành mạnh. Người bệnh cần kiểm soát đường huyết thường xuyên và tuân thủ điều trị. Chế độ ăn kiêng và tập luyện là yếu tố quan trọng trong điều trị.  Phát hiện sớm giúp kiểm soát bệnh hiệu quả và tránh biến chứng.',
            'Stress và căng thẳng trong công việc ảnh hưởng xấu đến sức khỏe tinh thần. Cần có thời gian nghỉ ngơi và thư giãn để giảm căng thẳng.  Tập yoga và thiền định giúp cân bằng tâm trí hiệu quả. Nếu stress kéo dài nên đi khám bác sĩ chuyên khoa tâm thần.',
            'Tập thể dục đều đặn giúp cải thiện sức khỏe và tăng cường miễn dịch. Nên tập ít nhất 30 phút mỗi ngày với các bài tập phù hợp. Chạy bộ bơi lội và đạp xe là những môn thể dục tốt.  Tập thể dục còn giúp giảm cân và cải thiện tâm trạng.',
        ],
        'Chính trị': [
            'Quốc hội thông qua nhiều luật quan trọng trong kỳ họp vừa qua. Các đại biểu đã thảo luận sôi nổi về các dự án luật mới. Luật đất đai và luật nhà ở được sửa đổi để phù hợp thực tế. Những luật này sẽ có tác động lớn đến đời sống người dân.',
            'Chính phủ triển khai chương trình cải cách hành chính toàn diện. Mục tiêu là tạo môi trường thuận lợi cho doanh nghiệp và người dân. Thủ tục hành chính được đơn giản hóa và số hóa nhiều hơn. Người dân có thể giải quyết hồ sơ trực tuyến tiết kiệm thời gian.',
            'Quan hệ ngoại giao Việt Nam với các nước ngày càng được củng cố. Nhiều chuyến thăm cấp cao được thực hiện tạo cơ hội hợp tác. Việt Nam tích cực tham gia các tổ chức và diễn đàn quốc tế. Vai trò của Việt Nam trên trường quốc tế ngày càng được nâng cao.',
            'Đảng Cộng sản Việt Nam tổ chức đại hội toàn quốc lần thứ mười ba. Đại hội bầu ra ban chấp hành trung ương khóa mới. Các nghị quyết quan trọng về phát triển kinh tế xã hội được thông qua. Đại hội là sự kiện chính trị quan trọng nhất của đất nước.',
            'Chính sách an sinh xã hội được Chính phủ quan tâm đặc biệt. Nhiều chương trình hỗ trợ người nghèo và người có hoàn cảnh khó khăn. Trợ cấp xã hội được tăng lên để đảm bảo đời sống người dân. Xây dựng xã hội công bằng và văn minh là mục tiêu lâu dài.',
        ],
        'Pháp luật': [
            'Bộ luật hình sự được sửa đổi bổ sung nhiều điều khoản mới. Hình phạt đối với tội phạm ma túy và tham nhũng được tăng nặng. Luật sư cho rằng đây là bước tiến quan trọng trong cải cách tư pháp. Người dân cần nắm rõ pháp luật để bảo vệ quyền lợi của mình.',
            'Tòa án xét xử vụ án tham nhũng lớn liên quan nhiều quan chức. Bị cáo bị cáo buộc tham ô hàng trăm tỷ đồng tiền nhà nước. Phiên tòa diễn ra công khai với sự theo dõi của dư luận. Bản án nghiêm khắc thể hiện quyết tâm chống tham nhũng của Nhà nước.',
            'Luật bảo vệ quyền lợi người tiêu dùng được ban hành giúp người dân. Người tiêu dùng có quyền khiếu nại khi mua phải hàng kém chất lượng. Doanh nghiệp vi phạm sẽ bị xử phạt và buộc bồi thường thiệt hại. Ý thức bảo vệ quyền lợi người tiêu dùng ngày càng được nâng cao.',
            'Tranh chấp đất đai giữa các hộ dân được tòa án giải quyết. Cả hai bên đều xuất trình giấy tờ chứng minh quyền sở hữu. Tòa án căn cứ vào pháp luật và chứng cứ để ra phán quyết. Việc giải quyết tranh chấp cần đảm bảo công bằng và minh bạch.',
            'Luật giao thông đường bộ có nhiều quy định về xử phạt vi phạm.  Lái xe sau khi uống rượu bia bị phạt rất nặng và tước bằng lái.  Camera giám sát ghi nhận vi phạm và gửi phạt nguội về nhà. Người dân cần chấp hành nghiêm luật giao thông để đảm bảo an toàn.',
        ],
        'Khoa học': [
            'Các nhà khoa học phát hiện ra loại virus mới có khả năng lây lan nhanh.  Nghiên cứu đang được tiến hành để tìm ra vaccine phòng ngừa hiệu quả. Cộng đồng quốc tế hợp tác chặt chẽ trong việc nghiên cứu và chia sẻ thông tin. Hy vọng sớm có được phương pháp điều trị và phòng ngừa tốt nhất.',
            'Kính viễn vọng không gian James Webb gửi về những hình ảnh vũ trụ tuyệt đẹp. Các nhà khoa học đang phân tích dữ liệu để tìm hiểu về nguồn gốc vũ trụ. Những phát hiện mới có thể thay đổi quan điểm về sự hình thành thiên hà. Đây là bước tiến lớn trong lĩnh vực thiên văn học hiện đại.',
            'Biến đổi khí hậu đang tác động nghiêm trọng đến môi trường toàn cầu. Nhiệt độ trái đất tăng cao gây ra nhiều hiện tượng thời tiết cực đoan. Các quốc gia cần hợp tác để giảm phát thải khí nhà kính. Khoa học công nghệ đóng vai trò quan trọng trong việc ứng phó biến đổi khí hậu.',
            'Nghiên cứu gen người giúp phát hiện và điều trị nhiều bệnh di truyền. Công nghệ chỉnh sửa gen CRISPR mở ra cơ hội chữa bệnh hiệu quả. Tuy nhiên việc can thiệp vào gen người cũng đặt ra nhiều vấn đề đạo đức. Cần có quy định pháp luật chặt chẽ để kiểm soát công nghệ này.',
            'Năng lượng tái tạo đang được đầu tư phát triển mạnh mẽ trên toàn thế giới.  Điện mặt trời và điện gió ngày càng trở nên phổ biến và rẻ hơn. Việt Nam cũng đang xây dựng nhiều dự án năng lượng tái tạo quy mô lớn. Đây là giải pháp bền vững cho vấn đề năng lượng và môi trường.',
        ],
        'Văn hóa': [
            'Lễ hội truyền thống đầu xuân được tổ chức tại nhiều địa phương trên cả nước. Người dân tham gia các hoạt động văn hóa dân gian phong phú đa dạng. Lễ hội là dịp để gìn giữ và phát huy bản sắc văn hóa dân tộc. Du khách trong và ngoài nước rất quan tâm và tham gia đông đảo.',
            'Bảo tàng mỹ thuật tổ chức triển lãm tranh của các họa sĩ nổi tiếng. Các tác phẩm nghệ thuật thể hiện phong cách độc đáo và sáng tạo. Triển lãm thu hút đông đảo người yêu nghệ thuật đến tham quan. Đây là cơ hội để công chúng tiếp cận với nghệ thuật đương đại.',
            'Di sản văn hóa Việt Nam được UNESCO công nhận là di sản thế giới. Việc bảo tồn và phát huy giá trị di sản được đặc biệt chú trọng. Các công trình di tích lịch sử được tu bổ và bảo vệ nghiêm ngặt. Du lịch văn hóa phát triển mạnh nhờ vào các di sản quý giá này.',
            'Âm nhạc truyền thống đang được giới trẻ quan tâm và học tập trở lại. Các nghệ nhân ưu tú truyền dạy nghề cho thế hệ sau một cách tận tâm. Nhiều chương trình biểu diễn âm nhạc dân tộc được tổ chức thu hút khán giả. Việc gìn giữ âm nhạc truyền thống là trách nhiệm của mọi thế hệ.',
            'Ngành xuất bản phát hành nhiều đầu sách mới về văn học và lịch sử. Các tác giả trẻ ngày càng có nhiều tác phẩm chất lượng được đánh giá cao. Thói quen đọc sách đang dần được khôi phục trong cộng đồng. Sách là nguồn tri thức quý giá giúp nâng cao hiểu biết của con người.',
        ],
        'Thời sự': [
            'Thủ tướng Chính phủ chủ trì họp bàn về kế hoạch phát triển kinh tế năm tới. Các bộ ngành báo cáo tình hình thực hiện nhiệm vụ trong năm vừa qua.  Chính phủ đặt mục tiêu tăng trưởng cao và bền vững cho đất nước. Nhiều chính sách mới sẽ được ban hành để hỗ trợ doanh nghiệp và người dân.',
            'Bão lớn đổ bộ vào miền Trung gây thiệt hại nặng nề về người và của.  Chính quyền địa phương đã sơ tán dân đến nơi an toàn trước khi bão đến. Lực lượng cứu hộ đang khẩn trương tìm kiếm người mất tích và hỗ trợ dân.  Cả nước chung tay ủng hộ đồng bào miền Trung vượt qua khó khăn.',
            'Giá xăng dầu tăng cao ảnh hưởng đến chi phí sinh hoạt của người dân. Chính phủ đang xem xét các biện pháp hỗ trợ để giảm gánh nặng cho dân. Nguyên nhân là do giá dầu thế giới biến động phức tạp không lường trước. Người dân cần tiết kiệm và sử dụng năng lượng hiệu quả hơn.',
            'Đường sắt đô thị chính thức đi vào hoạt động phục vụ người dân thành phố. Tuyến metro hiện đại giúp giảm tải ách tắc giao thông đáng kể. Người dân rất phấn khởi và hào hứng trải nghiệm phương tiện mới. Dự án này là thành quả của nhiều năm đầu tư xây dựng công phu.',
            'Hội nghị quốc tế về biến đổi khí hậu được tổ chức tại Hà Nội. Đại diện các nước thảo luận về giải pháp ứng phó với biến đổi khí hậu. Việt Nam cam kết giảm phát thải và chuyển đổi sang năng lượng sạch. Sự kiện nhận được sự quan tâm đặc biệt từ cộng đồng quốc tế.',
        ],
        'Thời sự 2': [  # Thêm dữ liệu để đủ số lượng
            'Lễ kỷ niệm ngày Quốc khánh được tổ chức trọng thể tại Quảng trường Ba Đình. Các đại biểu và nhân dân tham dự buổi lễ với niềm tự hào dân tộc. Lãnh đạo Đảng và Nhà nước phát biểu về những thành tựu đạt được.  Chương trình văn nghệ chào mừng rất hoành tráng và ý nghĩa.',
            'Chiến dịch vệ sinh môi trường được phát động trên toàn quốc vào cuối tuần. Hàng ngàn người tham gia dọn dẹp đường phố công viên và bãi biển. Ý thức bảo vệ môi trường của người dân ngày càng được nâng cao. Môi trường xanh sạch đẹp là trách nhiệm của mọi người.',
            'Ngân hàng Nhà nước công bố chính sách tiền tệ mới cho giai đoạn tới. Lãi suất được điều chỉnh để phù hợp với tình hình kinh tế hiện tại. Các chuyên gia nhận định đây là quyết định đúng đắn và kịp thời. Chính sách này sẽ ảnh hưởng đến hoạt động của cả nền kinh tế.',
            'Công an triệt phá đường dây buôn lậu xăng dầu quy mô lớn.  Nhiều đối tượng liên quan đã bị bắt giữ và xử lý theo pháp luật. Tang vật thu giữ gồm hàng trăm nghìn lít xăng dầu nhập lậu. Hành vi buôn lậu gây thiệt hại lớn cho ngân sách nhà nước.',
            'Chương trình thiện nguyện mang trung thu đến với trẻ em vùng cao. Các tình nguyện viên trao tặng quà và tổ chức vui chơi cho các em. Nụ cười hạnh phúc của trẻ em là động lực cho các nhà hảo tâm. Hoạt động từ thiện góp phần xây dựng xã hội tốt đẹp hơn.',
        ],
    }
    
    # Tạo DataFrame
    data_list = []
    for topic, texts in sample_data.items():
        # Bỏ số đếm trong tên topic
        topic_clean = topic.replace(' 2', '')
        for text in texts:
            data_list.append({'text': text, 'topic': topic_clean})
    
    df = pd.DataFrame(data_list)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Lưu file
    output_path = Path('data/processed_dataset.csv')
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Đã tạo {len(df)} mẫu dữ liệu")
    print(f"📊 Phân bố:")
    print(df['topic'].value_counts())
    print(f"\n💾 Lưu tại: {output_path}")
    
    return df

if __name__ == "__main__":
    print("="*60)
    print("TẠO DATASET MẪU CHO DỰ ÁN")
    print("="*60)
    print("\n⚠️  Dataset VNTC gặp vấn đề, sử dụng dữ liệu mẫu thay thế")
    print("📝 Dữ liệu mẫu gồm 50 văn bản cho 10 topics\n")
    
    df = create_sample_dataset()
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH!  ")
    print("="*60)
    print("\n➡️  Tiếp theo: Chạy train_model.py để train model")