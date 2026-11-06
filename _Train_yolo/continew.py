import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# Đường dẫn tới last.pt của lần train bị tạm dừng
LAST_PT = Path("C:\\Users\\ThisPC\\Desktop\\testapp\\runs\\pose\\runs\\pose\\ngontay_yolo26s_pose_train\\weights\\last.pt")
NEW_EPOCHS = 55

if __name__ == "__main__":
    if not LAST_PT.is_file():
        raise FileNotFoundError(f"Không thấy checkpoint: {LAST_PT.resolve()}")

    # Sao lưu trước khi sửa
    backup = LAST_PT.with_name("last_backup.pt")
    if not backup.exists():
        shutil.copy2(LAST_PT, backup)

    ckpt = torch.load(LAST_PT, map_location="cpu", weights_only=False)
    done_epoch = ckpt.get("epoch", -1) + 1  # số epoch đã train xong
    print(f"[*] Đã train xong {done_epoch} epoch, epochs cũ = {ckpt['train_args']['epochs']}")

    if done_epoch >= NEW_EPOCHS:
        print(f"[!] Đã train >= {NEW_EPOCHS} epoch, không cần train tiếp. "
              f"Dùng weights/best.pt luôn.")
    else:
        ckpt["train_args"]["epochs"] = NEW_EPOCHS
        torch.save(ckpt, LAST_PT)
        print(f"[*] Đã đổi epochs thành {NEW_EPOCHS}, bắt đầu resume...")

        model = YOLO(str(LAST_PT))
        model.train(resume=True)
        print("[*] Quá trình huấn luyện Pose hoàn tất!")