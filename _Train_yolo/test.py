import os
import torch
# Phải đặt TRƯỚC khi import paddle / paddleocr
os.environ["FLAGS_use_mkldnn"] = "0"                      # tránh lỗi oneDNN/PIR trên CPU
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import argparse
import time

import cv2
import paddle
from paddleocr import PaddleOCR


def detect_device(choice: str) -> str:
    """Trả về 'gpu:0' hoặc 'cpu' tuỳ lựa chọn và phần cứng thực tế."""
    has_gpu = False
    try:
        has_gpu = (
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:
        pass

    if choice == "cpu":
        return "cpu"
    if choice == "gpu":
        if has_gpu:
            return "gpu:0"
        print("[CẢNH BÁO] Không tìm thấy GPU/paddlepaddle-gpu -> chuyển sang CPU.")
        return "cpu"
    # auto
    return "gpu:0" if has_gpu else "cpu"


def load_ocr(device: str, fast: bool) -> PaddleOCR:
    kwargs = dict(
        lang="en",
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine_config={"run_mode": "paddle"},
    )
    if device == "cpu":
        kwargs["enable_mkldnn"] = False   # QUAN TRỌNG: tắt oneDNN trên CPU
    if fast:
        kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
        kwargs["text_recognition_model_name"] = "en_PP-OCRv5_mobile_rec"
    return PaddleOCR(**kwargs)


def build_and_warmup(device: str, fast: bool, img):
    """Khởi tạo model + chạy warm-up. Trả về (ocr, device đang dùng)."""
    print(f"Đang tải model PaddleOCR trên [{device.upper()}]...")
    ocr = load_ocr(device, fast)
    ocr.predict(input=img)  # warm-up (lần đầu luôn chậm)
    print("Tải model thành công!\n")
    return ocr, device


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR 3.x - đọc số (CPU/GPU)")
    parser.add_argument("--image", default="testorc.jpg", help="Đường dẫn ảnh")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                        help="auto = tự chọn GPU nếu có")
    parser.add_argument("--fast", action="store_true",
                        help="Dùng model mobile (nhẹ, nhanh hơn)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Số lần chạy để lấy tốc độ trung bình")
    args = parser.parse_args()

    # 1. Đọc ảnh
    img = cv2.imread(args.image)
    if img is None:
        print(f"[LỖI ĐỌC ẢNH] Không tìm thấy hoặc không đọc được: '{args.image}'")
        return

    # 2. Khởi tạo model (GPU lỗi thì tự lùi về CPU)
    device = detect_device(args.device)
    try:
        ocr, device = build_and_warmup(device, args.fast, img)
    except Exception as e:
        if device != "cpu":
            print(f"[LỖI GPU] {e}\n-> Thử lại bằng CPU...\n")
            try:
                ocr, device = build_and_warmup("cpu", args.fast, img)
            except Exception as e2:
                print(f"[LỖI KHỞI TẠO] Không thể chạy trên CPU: {e2}")
                return
        else:
            print(f"[LỖI KHỞI TẠO] Không thể tải model PaddleOCR: {e}")
            return

    # 3. Đo tốc độ (trung bình nhiều lần, sau warm-up)
    times = []
    result = None
    try:
        for _ in range(max(1, args.runs)):
            start = time.perf_counter()
            result = ocr.predict(input=img)
            times.append((time.perf_counter() - start) * 1000)
    except Exception as e:
        print(f"[LỖI XỬ LÝ OCR] Quá trình nhận diện gặp sự cố: {e}")
        return

    avg_ms = sum(times) / len(times)
    min_ms = min(times)

    # 4. Hiển thị kết quả
    print("=" * 34)
    print("         KẾT QUẢ ĐỌC OCR")
    print("=" * 34)

    if not result:
        print("Không có kết quả trả về từ model.")
        print("=" * 34)
        return

    try:
        res = result[0]
        texts = [str(t) for t in res["rec_texts"]]
        scores = [float(s) for s in res["rec_scores"]]

        if not texts:
            print("Không tìm thấy ký tự nào trong ảnh.")
            print("=" * 34)
            return

        digits_only = "".join(c for t in texts for c in t if c.isdigit())
        confidence = sum(scores) / len(scores)

        print(f"Thiết bị             : {device.upper()}")
        print(f"Các dòng nhận diện   : {texts}")
        print(f"Chuỗi thô            : {' '.join(texts)}")
        print(f"Chuỗi sau khi lọc số : {digits_only if digits_only else '(Không có số nào)'}")
        print(f"Độ tin cậy trung bình: {confidence * 100:.2f}%")
        print(f"Tốc độ trung bình    : {avg_ms:.2f} ms ({len(times)} lần)")
        print(f"Tốc độ nhanh nhất    : {min_ms:.2f} ms")

    except (KeyError, IndexError, TypeError) as e:
        print(f"[LỖI PARSE KẾT QUẢ] Cấu trúc trả về không như mong đợi: {e}")
        print(f"Dữ liệu gốc: {result}")

    print("=" * 34)


if __name__ == "__main__":
    main()