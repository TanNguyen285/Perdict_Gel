"""mlp_predict.py - Model MLP 1D (ONNX): đặc trưng góc -> (độ dài, điện trở).

File này chỉ lo việc nạp model và suy luận. Không biết gì về pose hay cách tính góc;
đặc trưng nào đưa vào model do logic.py quyết định (xem FingerAnalyzer.features).
Chưa nạp model thì predict() trả (None, None).
"""
import numpy as np
import onnxruntime as ort

import moitruong as env

NAN = float("nan")


class MLPPredictor:
    def __init__(self):
        self.sess, self.name, self.path, self.last_error = None, None, None, ""

    @property
    def loaded(self):
        return self.sess is not None

    def load(self, path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        self.path = path

    def reset(self):
        """Đặt lại trạng thái nội bộ khi bắt đầu phiên mới. Hiện MLP chưa có trạng thái nên không làm gì;
        để sẵn chỗ này cho bản MLP cửa sổ / có nhớ sau này."""

    def predict(self, feats):
        if not self.loaded:
            return None, None
        try:
            x = np.asarray(feats, dtype=np.float32).reshape(1, -1)
            if env.MLP_X_MEAN is not None and env.MLP_X_STD is not None:
                x = (x - np.asarray(env.MLP_X_MEAN, np.float32)) / np.asarray(env.MLP_X_STD, np.float32)
            y = np.asarray(self.sess.run(None, {self.name: x})[0], dtype=np.float64).reshape(-1)
            if env.MLP_Y_MEAN is not None and env.MLP_Y_STD is not None:
                y = y * np.asarray(env.MLP_Y_STD) + np.asarray(env.MLP_Y_MEAN)
            return float(y[0]), (float(y[1]) if y.size > 1 else NAN)
        except Exception as e:  # không để app sập vì MLP
            self.last_error = str(e)
            return None, None
