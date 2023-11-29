from collections import UserList
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Optional, Self, cast
import nptyping as npt

import cv2
import numpy as np
from sklearn.neighbors import BallTree

from .common_types import FaceDescriptor, FaceDescriptorList, NumpyImageRGB
from .id_holder import IdHolder
from .models import ArcFaceONNX, Scrfd
from .people_db import PeopleDatabase


class RecognitionLevel(Enum):
    MISMATCH = -1
    UNSURE = 0
    MATCH = 1

    def __lt__(self, other: Self) -> int:
        lesser: int = self.value.__lt__(other.value)
        return lesser

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RecognitionLevel):
            return False
        equal: bool = self.value.__eq__(other.value)
        return equal

    def __str__(self) -> str:
        return self.name


class VerificationResult:
    similarity: float

    def __str__(self) -> str:
        return str(self.level)

    def __repr__(self) -> str:
        return f'{self.level} ({self.similarity:.2f})'

    def __init__(self, sim: float) -> None:
        self.similarity = sim

    @property
    def level(self) -> RecognitionLevel:  # sourcery skip: assign-if-exp
        if self.similarity < 1/5:
            return RecognitionLevel(-1)
        if self.similarity < 1/3:
            return RecognitionLevel(0)
        else:
            return RecognitionLevel(1)

    @classmethod
    def from_dist(cls, dist: float) -> Self:
        sim = cls.dist_to_sim(dist)
        return cls(sim)

    @classmethod
    def sim_to_dist(cls, sim: float) -> float:
        return 1. - sim

    @classmethod
    def dist_to_sim(cls, dist: float) -> float:
        return 1. - dist

    @property
    def dist(self) -> float:
        return self.sim_to_dist(self.similarity)


class RecognitionResult(UserList[tuple[str, VerificationResult]]):
    def __str__(self) -> str:
        s = []
        for name, result in self.data:
            s.append(f'{name} - {result}')
        return ', '.join(s)


class RecognitionEngine:
    people_db: PeopleDatabase
    ids: IdHolder
    face_descriptor: ArcFaceONNX
    face_detector: Scrfd
    knn: BallTree

    SAMPLES_PER_PERSON = 3

    def __init__(self, db_dir: str | PathLike[str]) -> None:
        self.people_db = PeopleDatabase(db_dir)
        self.ids = IdHolder()

        self._onnx_setup()
        self.face_detector = Scrfd(
            (self.onnx_assets_dir / 'det_10g.onnx').as_posix()
        ).prepare(0)
        self.face_descriptor = ArcFaceONNX(
            (self.onnx_assets_dir / 'w600k_r50.onnx').as_posix()
        ).prepare(0)

        self._load_descriptors()

    def _onnx_setup(self) -> None:
        import onnxruntime
        onnxruntime.set_default_logger_severity(3)

        self.onnx_assets_dir = Path(
            '~/.insightface/models/buffalo_l').expanduser()

    def _read_image(self, img_path: str | PathLike[str]) -> NumpyImageRGB:
        img: NumpyImageRGB = np.array(cv2.imread(
            Path(img_path).as_posix(), cv2.IMREAD_COLOR))
        return img

    def _get_image(self, img_input: str | PathLike[str] | NumpyImageRGB) -> NumpyImageRGB:
        if isinstance(img_input, str | PathLike):
            return self._read_image(img_input)
        npt.assert_isinstance(img_input, NumpyImageRGB)
        return img_input

    def _get_descriptor_from_img(self, img: NumpyImageRGB) -> FaceDescriptor:
        _, kpss = self.face_detector.autodetect(img, max_num=1)
        kpss = cast(FaceDescriptorList, kpss)
        if len(kpss) < 1:
            raise RuntimeError("Could not find face in image")
        if len(kpss) > 1:
            raise RuntimeError("Multiple faces were found on image")
        return self.face_descriptor.get(img, kpss[0])

    def _load_descriptors(self) -> None:
        """Reads all images in `people_db` and
        loads its respective face descriptors into KNN data structure"""
        knn_X: list[FaceDescriptor] = []
        knn_Y: list[int] = []
        for person_name, person_img_paths in self.people_db.people.items():
            person_id = self.ids[person_name]
            for img_path in person_img_paths:
                person_descriptor = self._get_descriptor_from_img(
                    self._read_image(img_path))
                knn_X.append(person_descriptor)
                knn_Y.append(person_id)

        self.ball_tree = BallTree(
            data=knn_X,
            metric=lambda u, v: VerificationResult.sim_to_dist(
                self.face_descriptor.compute_sim(u, v))
        )
        self.knn_Y = knn_Y

    def query_image(
        self,
        img_input: str | PathLike[str] | NumpyImageRGB,
        num_results: Optional[int] = None
    ) -> RecognitionResult:
        """Returns best verification results for `img` against persons in `people_db`"""
        img = self._get_image(img_input)

        if not isinstance(num_results, int):
            k = self.SAMPLES_PER_PERSON
        else:
            k = np.clip(num_results, 0, self.SAMPLES_PER_PERSON)

        query_descriptor = self._get_descriptor_from_img(img)
        distances, indices = self.ball_tree.query(X=[query_descriptor], k=k)

        distances = distances.flatten()
        indices = indices.flatten()

        idx = np.argsort(distances)
        distances = np.take(distances, idx)
        indices = np.take(indices, idx)

        similarities = [
            VerificationResult.from_dist(d)
            for d in distances]
        person_ids = [
            self.ids.get_reverse(np.take(self.knn_Y, i))
            for i in indices]

        return RecognitionResult(zip(
            cast(Iterable[str], person_ids),
            similarities))

    def query_people(
        self,
        db: PeopleDatabase,
        num_results: Optional[int] = None
    ) -> dict[str, RecognitionResult]:
        results: dict[str, RecognitionResult] = {}
        for person_name, person_img_paths in db.people.items():
            print(f'Querying {person_name} ({self.ids[person_name]})')
            for img_path in person_img_paths:
                results[str(img_path)] = self.query_image(
                    img_path, num_results=num_results)

        return results
