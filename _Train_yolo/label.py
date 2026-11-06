import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

try:
    from PIL import Image
except ImportError:  # Pillow đi kèm ultralytics, nhưng phòng hờ
    Image = None


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "ngon tay"
DATASET_YAML = PROJECT_DIR / "dataset.yaml"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

BOX_CLASS = "man"         # chỉ có box, keypoint = 0 0 0
KEYPOINT_CLASS = "cogel"  # chỉ lấy các điểm keypoint_N; box tự sinh, box vẽ tay bị bỏ qua
FINGER_BOX_PX = 160       # cạnh box vuông tự sinh quanh 3 điểm (pixel ảnh gốc)
REQUIRE_ALL_KEYPOINTS = True  # True: ảnh không đủ keypoint thì loại khỏi dataset

BOX_SHAPE_TYPES = ("rectangle", "polygon")

KP_LABEL_RE = re.compile(r"^(?:keypoint|kp|point)[_\s-]*(\d+)$")


def norm_label(label):
    return label.strip().lower() if isinstance(label, str) else label


def keypoint_index(label):
    """'keypoint_2' -> 2 (1-based). Không phải keypoint -> None."""
    if not isinstance(label, str):
        return None
    m = KP_LABEL_RE.match(label.strip().lower())
    return int(m.group(1)) if m else None


def load_dataset_config():
    with DATASET_YAML.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset configuration is not a YAML mapping: {DATASET_YAML}")

    configured_root = config.get("path")
    if not configured_root:
        raise ValueError(f"Missing 'path' in {DATASET_YAML}")
    dataset_root = Path(configured_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = DATASET_YAML.parent / dataset_root
    dataset_root = dataset_root.resolve()

    train_path, val_path = config.get("train"), config.get("val")
    if not isinstance(train_path, str) or not isinstance(val_path, str):
        raise ValueError("dataset.yaml must define train and val as directory paths.")
    train_dir = (dataset_root / train_path).resolve()
    val_dir = (dataset_root / val_path).resolve()
    if train_dir == val_dir:
        raise ValueError("dataset.yaml train and val paths must be different.")

    def label_dir(image_dir):
        try:
            rel = image_dir.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError("Train and val paths must be inside the dataset root.") from error
        if not rel.parts or rel.parts[0] != "images":
            raise ValueError("Train and val paths must be inside an 'images' directory.")
        return dataset_root / "labels" / Path(*rel.parts[1:])

    kpt_shape = config.get("kpt_shape")
    if (
        not isinstance(kpt_shape, list)
        or len(kpt_shape) != 2
        or not all(isinstance(v, int) and v > 0 for v in kpt_shape)
    ):
        raise ValueError("dataset.yaml must define kpt_shape as [keypoints, dimensions].")
    if kpt_shape[1] != 3:
        raise ValueError(
            "kpt_shape phải là [N, 3] (x, y, visibility) để class 'man' "
            "có visibility = 0 và bị bỏ qua khỏi keypoint loss."
        )

    names = config.get("names")
    if isinstance(names, list):
        class_names = dict(enumerate(names))
    elif isinstance(names, dict):
        class_names = {int(k): v for k, v in names.items()}
    else:
        raise ValueError("dataset.yaml must define class names as a list or mapping.")
    if not class_names or not all(isinstance(n, str) for n in class_names.values()):
        raise ValueError("dataset.yaml contains invalid class names.")
    for required in (BOX_CLASS, KEYPOINT_CLASS):
        if required not in class_names.values():
            raise ValueError(f"Class {required!r} không có trong dataset.yaml names.")

    return (
        dataset_root,
        train_dir,
        val_dir,
        label_dir(train_dir),
        label_dir(val_dir),
        class_names,
        kpt_shape,
    )


def find_image(json_file, image_path):
    if image_path:
        image_file = SOURCE_DIR / Path(image_path).name
        if image_file.is_file():
            return image_file
    for ext in IMAGE_EXTENSIONS:
        image_file = json_file.with_suffix(ext)
        if image_file.is_file():
            return image_file
    return None


def actual_image_size(image_file):
    """Kích thước ảnh sau khi áp EXIF (giống cách Ultralytics đọc)."""
    if Image is None:
        return None
    try:
        with Image.open(image_file) as im:
            w, h = im.size
            orientation = im.getexif().get(274)
            if orientation in (5, 6, 7, 8):
                w, h = h, w
            return w, h
    except Exception:
        return None


def clamp01(v):
    return min(max(v, 0.0), 1.0)


def parse_points(points, min_count):
    """Trả về list (x, y) pixel, hoặc None nếu dữ liệu lỗi."""
    if not isinstance(points, list) or len(points) < min_count:
        return None
    parsed = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return None
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        parsed.append((x, y))
    return parsed


def fmt(values):
    return " ".join(str(v) if isinstance(v, int) else f"{v:.6f}" for v in values)


def bbox_norm(pts, img_w, img_h):
    """(x1, y1, x2, y2) chuẩn hoá 0-1 từ list điểm pixel."""
    xs = [clamp01(x / img_w) for x, _ in pts]
    ys = [clamp01(y / img_h) for _, y in pts]
    return min(xs), min(ys), max(xs), max(ys)


def make_screen_line(class_id, shape, img_w, img_h, kpt_count):
    """'man': box từ rectangle/polygon (lấy min/max mọi điểm), keypoint = 0 0 0."""
    pts = parse_points(shape.get("points"), 2)
    if pts is None:
        return None
    x1, y1, x2, y2 = bbox_norm(pts, img_w, img_h)
    w, h = x2 - x1, y2 - y1
    if w <= 1e-6 or h <= 1e-6:
        return None

    values = [class_id, (x1 + x2) / 2, (y1 + y2) / 2, w, h]
    values += [0, 0, 0] * kpt_count
    return fmt(values)


def collect_keypoints(shapes, kpt_count, json_name):
    """Lấy mọi shape keypoint_N trong ảnh, KHÔNG phụ thuộc box hay group_id.

    Mỗi ảnh có 1 ngón -> gộp tất cả điểm thành 1 instance {idx(1-based): (x, y)}.
    Nếu một index xuất hiện nhiều lần (nhiều ngón) -> tách theo group_id và cảnh báo.
    Trả về list instance.
    """
    items = []
    for s in shapes:
        idx = keypoint_index(s.get("label"))
        if idx is None:
            continue
        if not (1 <= idx <= kpt_count):
            print(f"[WARN] {json_name}: {s.get('label')!r} ngoài khoảng 1..{kpt_count}, bỏ qua")
            continue
        pts = parse_points(s.get("points"), 1)
        if pts is None:
            print(f"[WARN] {json_name}: toạ độ lỗi ở {s.get('label')!r}")
            continue
        items.append((idx, pts[0][0], pts[0][1], s.get("group_id")))

    if not items:
        return []
    idxs = [i for i, _, _, _ in items]
    if len(set(idxs)) == len(idxs):
        return [{i: (x, y) for i, x, y, _ in items}]

    print(f"[WARN] {json_name}: có keypoint trùng index (nhiều ngón?) -> tách theo group_id")
    groups = {}
    for i, x, y, gid in items:
        groups.setdefault(gid, {}).setdefault(i, (x, y))
    return list(groups.values())


def make_finger_line(class_id, kps, img_w, img_h, kpt_count):
    """cogel: keypoint thật (visibility=2), box vuông cố định quanh tâm các điểm.

    Điểm thiếu -> 0 0 0. Box luôn chứa mọi keypoint.
    """
    if not kps:
        return None
    norm = {i: (clamp01(x / img_w), clamp01(y / img_h)) for i, (x, y) in kps.items()}
    cx = sum(p[0] for p in norm.values()) / len(norm)
    cy = sum(p[1] for p in norm.values()) / len(norm)

    half_w = FINGER_BOX_PX / img_w / 2
    half_h = FINGER_BOX_PX / img_h / 2
    x1 = clamp01(min(cx - half_w, min(p[0] for p in norm.values())))
    x2 = clamp01(max(cx + half_w, max(p[0] for p in norm.values())))
    y1 = clamp01(min(cy - half_h, min(p[1] for p in norm.values())))
    y2 = clamp01(max(cy + half_h, max(p[1] for p in norm.values())))

    values = [class_id, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    for i in range(1, kpt_count + 1):
        if i in norm:
            values += [norm[i][0], norm[i][1], 2]
        else:
            values += [0, 0, 0]
    return fmt(values)


def clear_dir(directory):
    """Xóa ảnh/label cũ trong thư mục train/val để không bị sót hay trùng."""
    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in (*IMAGE_EXTENSIONS, ".txt", ".cache"):
            f.unlink()


def main():
    (
        dataset_root,
        train_image_dir,
        val_image_dir,
        train_label_dir,
        val_label_dir,
        class_names,
        (kpt_count, _),
    ) = load_dataset_config()
    class_ids = {norm_label(name): cid for cid, name in class_names.items()}

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục nguồn: {SOURCE_DIR}")

    out_dirs = (train_image_dir, val_image_dir, train_label_dir, val_label_dir)
    src = SOURCE_DIR.resolve()
    for d in out_dirs:
        if src == d or d in src.parents:
            raise ValueError(f"SOURCE_DIR không được nằm trong {d} (sẽ bị xóa sạch).")

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        clear_dir(d)
    for name in ("train.cache", "val.cache"):
        (dataset_root / "labels" / name).unlink(missing_ok=True)

    json_files = sorted(SOURCE_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"Không có file .json nào trong: {SOURCE_DIR}")

    samples, skipped = [], 0
    class_count = Counter()
    print(f"Found {len(json_files)} JSON annotation files in: {SOURCE_DIR}")
    print(f"YOLO dataset root: {dataset_root}")

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_file = find_image(json_file, data.get("imagePath"))
        if image_file is None:
            print(f"[SKIP] No matching image for annotation: {json_file.name}")
            skipped += 1
            continue

        img_w, img_h = data.get("imageWidth"), data.get("imageHeight")
        if (
            not isinstance(img_w, (int, float))
            or not isinstance(img_h, (int, float))
            or img_w <= 0
            or img_h <= 0
        ):
            print(f"[SKIP] Invalid image dimensions in: {json_file.name}")
            skipped += 1
            continue

        real_size = actual_image_size(image_file)
        if real_size and real_size != (int(img_w), int(img_h)):
            print(f"[WARN] {json_file.name}: JSON ghi {int(img_w)}x{int(img_h)} "
                  f"nhưng ảnh thật là {real_size[0]}x{real_size[1]} (có thể do EXIF xoay)")

        shapes = data.get("shapes", [])

        lines, found, local_count = [], set(), Counter()
        for shape in shapes:
            raw_label = shape.get("label")
            label = norm_label(raw_label)
            # keypoint_N được gom riêng; box cogel vẽ tay/auto không dùng
            if keypoint_index(raw_label) is not None or label == KEYPOINT_CLASS:
                continue

            class_id = class_ids.get(label)
            if class_id is None:
                print(f"[SKIP] Unknown class {raw_label!r} in {json_file.name}; "
                      f"valid: {list(class_names.values())}")
                continue

            shape_type = shape.get("shape_type")
            if shape_type not in BOX_SHAPE_TYPES:
                print(f"[SKIP] {json_file.name}: {label!r} phải là rectangle, "
                      f"đang là {shape_type!r}")
                continue
            line = make_screen_line(class_id, shape, img_w, img_h, kpt_count)
            if line is None:
                print(f"[SKIP] Invalid points in {json_file.name} ({label})")
                continue
            lines.append(line)
            found.add(label)
            local_count[label] += 1

        # cogel: chỉ từ các điểm keypoint_N
        instances = collect_keypoints(shapes, kpt_count, json_file.name)
        incomplete = [
            [i for i in range(1, kpt_count + 1) if i not in inst] for inst in instances
        ]
        incomplete = [m for m in incomplete if m]
        if not instances or incomplete:
            reason = ("không có keypoint nào" if not instances
                      else "thiếu keypoint " + ", ".join(str(m) for m in incomplete))
            if REQUIRE_ALL_KEYPOINTS:
                print(f"[SKIP] {json_file.name}: {reason} -> loại ảnh khỏi dataset")
                skipped += 1
                continue
            print(f"[WARN] {json_file.name}: {reason}")
        cogel_id = class_ids[KEYPOINT_CLASS]
        for inst in instances:
            line = make_finger_line(cogel_id, inst, img_w, img_h, kpt_count)
            if line:
                lines.append(line)
                found.add(KEYPOINT_CLASS)
                local_count[KEYPOINT_CLASS] += 1

        if not lines:
            print(f"[SKIP] No usable labeled shapes in: {json_file.name}")
            skipped += 1
            continue

        if BOX_CLASS not in found:
            print(f"[WARN] {json_file.name}: thiếu {BOX_CLASS!r}")

        class_count.update(local_count)
        samples.append((image_file, "\n".join(lines) + "\n"))

    random.Random(RANDOM_SEED).shuffle(samples)
    if len(samples) > 1:
        train_count = min(max(int(len(samples) * TRAIN_RATIO), 1), len(samples) - 1)
    else:
        train_count = len(samples)

    splits = (
        (samples[:train_count], train_image_dir, train_label_dir),
        (samples[train_count:], val_image_dir, val_label_dir),
    )
    for split_samples, image_dir, label_dir in splits:
        for image_file, label_text in split_samples:
            shutil.copy2(image_file, image_dir / image_file.name)
            (label_dir / f"{image_file.stem}.txt").write_text(label_text, encoding="utf-8")

    print(f"Exported labeled images: {len(samples)}")
    print(f"Train images: {train_count}")
    print(f"Validation images: {len(samples) - train_count}")
    print(f"Skipped annotations: {skipped}")
    print(f"Boxes per class: {dict(class_count)}")
    print(f"Dataset YAML: {DATASET_YAML}")


if __name__ == "__main__":
    main()