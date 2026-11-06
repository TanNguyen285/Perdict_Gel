import os
from ultralytics import YOLO

if __name__ == "__main__":
  # Đường dẫn tới file weights đã train xong
  best_model_path = r"C:\Users\ThisPC\Desktop\testapp\runs\pose\runs\pose\ngontay_yolo26s_pose_train\weights\best.pt"

  if not os.path.exists(best_model_path):
    print(f"[Lỗi] Không tìm thấy file model tại: {best_model_path}")
  else:
    print(f"Đang load model từ: {best_model_path}")
    model = YOLO(best_model_path)

    print("Đang tiến hành xuất sang ONNX...")
    # Thêm tham số opset=11 hoặc opset=12 để tránh lỗi xung đột phiên bản
    success_path = model.export(format="onnx", imgsz=640, simplify=True, opset=12)
    print(f"[SUCCESS] Đã xuất file ONNX thành công tại: {success_path}")