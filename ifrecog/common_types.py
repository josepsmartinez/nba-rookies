from typing import Any

from nptyping import NDArray
from nptyping import Shape as NDShape
import nptyping as npt


NumpyArray = NDArray

NumpyMatrix = NDArray[NDShape['Channels, Height, Width'], Any]
NumpyImage = NDArray[NDShape['[b, g, r], Height, Width'], Any]
NumpyImageRGB = NDArray[NDShape['[b, g, r], Height, Width'], npt.UInt8]

FaceDescriptor = NDArray[NDShape['5, [x, y]'], npt.Float]
FaceDescriptorList = NDArray[NDShape['Faces, 5, [x, y]'], npt.Float]
