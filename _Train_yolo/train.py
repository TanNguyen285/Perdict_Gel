from ultralytics import YOLO

if __name__ == "__main__":
  # YOLO26 small Pose pretrained weights; Ultralytics downloads them if needed.
  model = YOLO("best.pt")

  # 2. Bắt đầu train mô hình
  results = model.train(
      data="dataset.yaml",  # Đường dẫn tới file yaml cấu hình dataset pose
      epochs=50,  # Số vòng lặp train
      imgsz=640,  # Kích thước ảnh đầu vào
      batch=32,  # Kích thước batch (tùy chỉnh theo VRAM của GPU)
      device=0,  # Chọn GPU (0) hoặc 'cpu' nếu không có card rời
      task="pose",  # Định nghĩa rõ tác vụ là pose estimation
      save=True,
      project="runs/pose",
      name="ngontay_yolo26s_pose_train",
      workers=0,  # Đặt workers=0 trên Windows để an toàn tuyệt đối với đa luồng
  )

  print("[*] Quá trình huấn luyện Pose hoàn tất!")
