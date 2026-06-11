# Dự án PBL4 - NGHIÊN CỨU VÀ ỨNG DỤNG HỌC SÂU ĐA PHƯƠNG THỨC TRONG PHÂN TÍCH CẢM XÚC CON NGƯỜI
Dự án này chứa mã nguồn cho ứng dụng giao diện (UI) và mô hình nhận diện.
bạn phải cấu hình đường dẫn ở trong các file này phù hợp với ổ đĩa của bạn.
## Hướng dẫn chạy code hệ thống nhận dạng cảm xúc đa phương thức 
1. Mở tệp Final_Mental_Health.py, bạn sẽ thấy cấu hình đường dẫn ở đầu source code.
2. Để chạy, cần cài thêm 2 mô hình 1. PhoWhisper để nhận dạng giọng nói và 2. Roberta. (tải tại link: https://drive.google.com/drive/folders/1g-3AZ2uB1vTHn3J-Yr0DB_lm1UTrvGhr)
3. Cấu hình lại đường dẫn sao cho phù hợp với ổ đĩa của bạn
4. Run and Enjoy

## Hướng dẫn tải clip đánh giá hệ thống đa phương thức từ RAVDESS 
1. Chạy Final_Tai_10DienVien_tu_RAVDESS.py, tệp này sẽ tự động tải 1040 clip từ https://zenodo.org/records/1188976 
2. Các clip được diễn viên nói bằng tiếng Anh, nếu muốn sửa thành Tiếng Việt, bạn phải chạy Final_ThemTiengVietVaoRAVDESS.py 
3. Để test hệ thống, bạn chỉ cần chạy Final_DanhGia60clip.py, hệ thống sẽ xuất kết quả cho bạn

## Một vài đường link quan trọng
1. Link Notebook huấn luyện ResNet trên RAF-DB ở đây : https://www.kaggle.com/code/trisleevawnminh/resnet18-t-ng-c-ng-ti-u-chu-n
