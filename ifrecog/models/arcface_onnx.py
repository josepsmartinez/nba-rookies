import os
from typing import IO, Optional, Self

import cv2
import numpy as np
import onnx
import onnxruntime
from skimage import transform as trans

from ..common_types import FaceDescriptor, NumpyArray, NumpyImage


src1: FaceDescriptor = np.array([
    [51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
    [51.157, 89.050], [57.025, 89.702]], dtype=np.float32)

src2: FaceDescriptor = np.array([
    [45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
    [45.177, 86.190], [64.246, 86.758]], dtype=np.float32)

src3: FaceDescriptor = np.array([
    [39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
    [42.463, 87.010], [69.537, 87.010]], dtype=np.float32)

src4: FaceDescriptor = np.array([
    [46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
    [48.167, 86.758], [67.236, 86.190]], dtype=np.float32)

src5: FaceDescriptor = np.array([
    [54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
    [55.388, 89.702], [61.257, 89.050]], dtype=np.float32)

# left-profile, left, frontal, right, right-profile
src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src: FaceDescriptor = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


def estimate_norm(
    lmk: NumpyArray, image_size: int = 112, mode: str = 'arcface'
) -> tuple[NumpyArray, int]:
    assert lmk.shape == (5, 2)
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M: NumpyArray
    min_index: int
    min_error = float('inf')
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(
    img: NumpyImage,
    landmark: NumpyArray,
    image_size: int = 112,
    mode: str = 'arcface'
) -> NumpyImage:
    M, pose_index = estimate_norm(landmark, image_size, mode)
    return np.array(
        cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0))


def square_crop(im: NumpyImage, S: int) -> tuple[NumpyImage, float]:
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(
    data: NumpyImage,
    center: tuple[int, int],
    output_size: int | float,
    scale: float,
    rotation: int | float
) -> tuple[NumpyImage, NumpyArray]:
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M


def trans_points2d(pts: NumpyArray, M: NumpyArray) -> NumpyArray:
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i] = new_pt[:2]

    return new_pts


def trans_points3d(pts: NumpyArray, M: NumpyArray) -> NumpyArray:
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        new_pts[i][:2] = new_pt[:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts: NumpyArray, M: NumpyArray) -> NumpyArray:
    return trans_points2d(pts, M) if pts.shape[1] == 2 else trans_points3d(pts, M)


class ArcFaceONNX:
    def __init__(
        self,
        model_file: str | IO[bytes],
        session: Optional[onnxruntime.InferenceSession] = None
    ) -> None:
        self.model_file = model_file
        if session is None:
            if isinstance(model_file, str):
                model_file = str(model_file)
                assert os.path.exists(model_file)
            self.session = onnxruntime.InferenceSession(model_file)
        else:
            self.session = None
        self.taskname = 'recognition'
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for node in graph.node[:8]:
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = [out.name for out in outputs]
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def prepare(self, ctx_id: int) -> Self:
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        return self

    def get(self, img: NumpyImage, kps: NumpyArray) -> FaceDescriptor:
        aimg = norm_crop(img, landmark=kps, image_size=self.input_size[0])
        return self.get_feat(aimg).flatten()

    def compute_sim(self, feat1: FaceDescriptor, feat2: FaceDescriptor) -> float:
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        return float(
            np.dot(feat1, feat2) / (norm(feat1) * norm(feat2)))

    def get_feat(self, imgs: list[NumpyImage] | NumpyImage) -> NumpyArray:
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(
            imgs, 1.0 / self.input_std, input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        return np.array(
            self.session.run(self.output_names, {self.input_name: blob})[0])

    def forward(self, batch_data: NumpyArray) -> NumpyArray:
        blob = (batch_data - self.input_mean) / self.input_std
        return np.array(
            self.session.run(self.output_names, {self.input_name: blob})[0])
