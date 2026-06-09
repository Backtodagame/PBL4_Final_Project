import os
import zipfile
import urllib.request

target_dir = r"E:\HK8\PBL4\Code\RAVDESS_Sub"
os.makedirs(target_dir, exist_ok=True)

# Duyệt vòng lặp tải từ Actor 01 đến Actor 10
for actor_id in range(1, 11):
    actor_name = f"Actor_{actor_id:02d}"
    
    # Kiểm tra nếu thư mục diễn viên đã tồn tại thì bỏ qua để tiết kiệm thời gian
    if os.path.exists(os.path.join(target_dir, actor_name)):
        print(f"⏩ {actor_name} đã tồn tại trên hệ thống, tự động bỏ qua.")
        continue
        
    zip_name = f"Video_Speech_{actor_name}.zip"
    zenodo_url = f"https://zenodo.org/records/1188976/files/{zip_name}"
    zip_path = os.path.join(target_dir, zip_name)
    
    try:
        print(f"\n📥 Đang tải tự động gói video chính diện của {actor_name}...")
        urllib.request.urlretrieve(zenodo_url, zip_path)
        print(f"✅ Tải {zip_name} thành công!")
        
        print(f"🔓 Đang giải nén dữ liệu {actor_name} vào thư mục hệ thống...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        # Dọn dẹp file zip thừa ngay sau khi giải nén thành công
        os.remove(zip_path)
        print(f"🧹 Đã xóa file zip tạm của {actor_name}.")
        
    except Exception as e:
        print(f"❌ Lỗi khi tải hoặc giải nén {actor_name}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

print("\n🎉 Hoàn tất quá trình mở rộng tập thực nghiệm! 10 diễn viên đã sẵn sàng chiến đấu.")