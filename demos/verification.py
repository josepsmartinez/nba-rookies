import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from ifrecog.models import ArcFaceONNX, Scrfd


onnxruntime.set_default_logger_severity(3)

assets_dir = os.path.expanduser('~/.insightface/models/buffalo_l')

detector = Scrfd(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)

rec = ArcFaceONNX(os.path.join(assets_dir, 'w600k_r50.onnx'))
rec.prepare(0)

ip1 = Path('.') / 'data/people/base/Victor Wembanyama/1.jpg'
image1 = cv2.imread(ip1.as_posix())

bboxes1, kpss1 = detector.autodetect(image1, max_num=1)
if bboxes1.shape[0] == 0:
    raise RuntimeError("Face not found in Image-1")
kps1 = np.array(kpss1)[0]
feat1 = rec.get(image1, kps1)

ip2 = Path('.') / 'data/people/base/Scoot Henderson/1.jpg'
image2 = cv2.imread(ip2.as_posix())
bboxes2, kpss2 = detector.autodetect(image2, max_num=1)
if bboxes2.shape[0] == 0:
    raise RuntimeError("Face not found in Image-2")
kps2 = np.array(kpss2)[0]
feat2 = rec.get(image2, kps2)
sim = rec.compute_sim(feat1, feat2)

print(f'Input images have the following cosine similarity: {sim:.3f}')
