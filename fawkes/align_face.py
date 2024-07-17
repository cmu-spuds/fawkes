import numpy as np
from mtcnn import MTCNN


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def aligner():
    return MTCNN(min_face_size=30)


def align(orig_img, aligner):
    """run MTCNN face detector"""

    if orig_img.ndim < 2:
        return None
    if orig_img.ndim == 2:
        orig_img = to_rgb(orig_img)
    orig_img = orig_img[:, :, 0:3]

    detect_results = aligner.detect_faces(orig_img)
    cropped_arr = []
    bounding_boxes_arr = []
    for dic in detect_results:
        if dic["confidence"] < 0.9:
            continue
        x, y, width, height = dic["box"]

        if width < 30 or height < 30:
            continue
        bb = [y, x, y + height, x + width]
        cropped = orig_img[bb[0] : bb[2], bb[1] : bb[3], :]
        cropped_arr.append(np.copy(cropped))
        bounding_boxes_arr.append(bb)

    return cropped_arr, bounding_boxes_arr
