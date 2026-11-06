"""Load YOLO-pose ONNX, decode keypoints and draw the target-class box."""
import ast
import cv2
import numpy as np
import onnxruntime as ort

import moitruong as env


def letterbox(img, new_hw, color=(114, 114, 114)):
    th, tw = new_hw
    h, w = img.shape[:2]
    r = min(th / h, tw / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (tw - nw) / 2, (th - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


class PoseONNX:
    """YOLO-pose ONNX trên CPU. Tự nhận 2 dạng output:
       - NMS-free (YOLO26):  [1, N, 6 + nk*3]  (x1,y1,x2,y2,conf,cls, kpts...)
       - Dạng cũ:            [1, 4+1+nk*3, 8400] (cx,cy,w,h,conf, kpts...)"""

    def __init__(self, path, nk=env.NUM_KPT, threads=env.ORT_THREADS):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        metadata = self.sess.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(metadata.get("names", "{}"))
        except (ValueError, SyntaxError) as e:
            raise ValueError("Metadata ONNX có names không hợp lệ.") from e
        self.names = {int(class_id): str(name) for class_id, name in names.items()}
        self.keypoint_class_id = next(
            (class_id for class_id, name in self.names.items()
             if name.casefold() == env.POSE_KEYPOINT_CLASS.casefold()),
            None,
        )
        if self.keypoint_class_id is None:
            raise ValueError(
                f"Model Pose không có class keypoint '{env.POSE_KEYPOINT_CLASS}'. "
                f"Các class hiện có: {', '.join(self.names.values()) or '(không có metadata)'}"
            )
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.fp16 = "float16" in inp.type
        s = inp.shape
        self.h = s[2] if isinstance(s[2], int) else env.INPUT_SIZE
        self.w = s[3] if isinstance(s[3], int) else env.INPUT_SIZE
        self.nk = nk
        # warm-up để lần chạy đầu không bị giật
        self.sess.run(None, {self.in_name: np.zeros((1, 3, self.h, self.w), np.float16 if self.fp16 else np.float32)})

    def predict(self, img, conf=0.25, iou=0.45, max_det=1):
        """Trả về (keypoint, score, box xyxy), với tọa độ ảnh gốc."""
        h0, w0 = img.shape[:2]
        lb, r, (px, py) = letterbox(img, (self.h, self.w))
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (self.w, self.h), swapRB=True, crop=False)
        if self.fp16:
            blob = blob.astype(np.float16)
        out = self.sess.run(None, {self.in_name: blob})[0]
        return self._decode(out, conf, iou, max_det, r, px, py, w0, h0)

    def _decode(self, out, conf, iou, max_det, r, px, py, w0, h0):
        p = np.asarray(out, dtype=np.float32)
        if p.ndim == 3:
            p = p[0]
        def detect_format(feature_count):
            if feature_count > 6 and (feature_count - 6) % 3 == 0:
                output_nk = (feature_count - 6) // 3
                if self.nk <= output_nk <= self.nk + 1:
                    return "e2e", output_nk
            if feature_count > 5 and (feature_count - 5) % 3 == 0:
                output_nk = (feature_count - 5) // 3
                if self.nk <= output_nk <= self.nk + 1:
                    return "old", output_nk
            return None

        detected = detect_format(p.shape[1])
        if detected is None:
            detected = detect_format(p.shape[0])
            if detected is None:
                raise ValueError(
                    f"Output ONNX không có đủ {self.nk} keypoint: shape {out.shape}"
                )
            p = p.T
        fmt, output_nk = detected

        scores = p[:, 4]
        class_ids = np.rint(p[:, 5]).astype(np.int64) if fmt == "e2e" else np.zeros(len(p), dtype=np.int64)
        m = scores >= conf
        p, scores = p[m], scores[m]
        class_ids = class_ids[m]
        if len(p) == 0:
            return []
        if fmt == "e2e":
            kp = p[:, 6:6 + output_nk * 3]
            selected = []
            for class_id in np.unique(class_ids):
                class_indices = np.flatnonzero(class_ids == class_id)
                selected.extend(class_indices[np.argsort(-scores[class_indices])[:max_det]])
            idx = np.asarray(selected, dtype=np.int64)
        else:
            b = p[:, :4].copy()
            b[:, 0] -= b[:, 2] / 2
            b[:, 1] -= b[:, 3] / 2
            keep = cv2.dnn.NMSBoxes(b.tolist(), scores.tolist(), conf, iou)
            idx = np.array(keep).reshape(-1)[:max_det]
            kp = p[:, 5:5 + output_nk * 3]
        if len(idx) == 0:
            return []
        res = []
        for i in idx:
            k = kp[i].reshape(output_nk, 3)[:self.nk].copy()
            k[:, 0] = np.clip((k[:, 0] - px) / r, 0, w0 - 1)
            k[:, 1] = np.clip((k[:, 1] - py) / r, 0, h0 - 1)
            if fmt == "e2e":
                box = p[i, :4].copy()
            else:
                cx, cy, bw, bh = p[i, :4]
                box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            box[[0, 2]] = np.clip((box[[0, 2]] - px) / r, 0, w0 - 1)
            box[[1, 3]] = np.clip((box[[1, 3]] - py) / r, 0, h0 - 1)
            class_name = self.names.get(int(class_ids[i]), str(int(class_ids[i])))
            res.append((k, float(scores[i]), box, class_name))
        return res


def _label(img, text, org, scale, color=(255, 255, 255)):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    x = max(2, min(x, img.shape[1] - tw - 6))
    y = max(th + 4, min(y, img.shape[0] - 4))
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + bl), env.LABEL_BG, -1)
    cv2.putText(img, text, (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_pose(img, kpts, fill=True):
    """Vẽ tam giác keypoint với nền đa giác trong suốt và các điểm đánh dấu."""
    h, w = img.shape[:2]
    s = max(h, w) / 640.0
    lw, rad = max(2, int(round(2 * s))), max(4, int(round(5 * s)))
    pts = kpts[:, :2].round().astype(np.int32)
    if fill and len(pts) >= 3:
        x, y, bw, bh = cv2.boundingRect(pts)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw + 1), min(h, y + bh + 1)
        if x1 > x0 and y1 > y0:
            roi = img[y0:y1, x0:x1]
            ov = roi.copy()
            cv2.fillPoly(ov, [pts - np.array([x0, y0], np.int32)], env.FILL_COLOR)
            cv2.addWeighted(ov, 0.4, roi, 0.6, 0, roi)
    cv2.polylines(img, [pts], True, env.LINE_COLOR, lw, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), rad, env.POINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), rad, env.POINT_EDGE, max(1, lw - 1), cv2.LINE_AA)


def draw_detections(img, detections):
    """Draw each detected class box with an ASCII label."""
    s = max(img.shape[:2]) / 640.0
    lw, fs = max(2, int(round(2 * s))), max(0.45, 0.5 * s)
    for _, score, box, class_name in detections:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), env.LINE_COLOR, lw, cv2.LINE_AA)
        safe_name = class_name.encode("ascii", "replace").decode("ascii")
        _label(img, f"{safe_name} {score:.2f}", (x1, y1 - int(8 * s)), fs)


def draw_result(img, res, fill=True):
    """Draw the keypoint triangle for the selected pose detection."""
    draw_pose(img, res.kpts, fill)
