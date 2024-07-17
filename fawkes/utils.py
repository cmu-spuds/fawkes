#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/
import gzip
import os
import pickle

import numpy as np
from keras import utils, backend
import tensorflow as tf
from tf_keras import models
from PIL import Image, ExifTags, UnidentifiedImageError
from fawkes.align_face import align

IMG_SIZE = 112
PREPROCESS = "raw"


def load_image(path):
    try:
        img = Image.open(path)
    except UnidentifiedImageError:
        return None
    except IsADirectoryError:
        return None

    try:
        info = img._getexif()
    except OSError:
        return None

    if info is not None:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        exif = dict(img._getexif().items())
        if orientation in exif.keys():
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
            else:
                pass
    img = img.convert("RGB")
    image_array = utils.img_to_array(img)

    return image_array


def filter_image_paths(image_paths):
    print("Identify {} files in the directory".format(len(image_paths)))
    new_image_paths = []
    new_images = []
    for p in image_paths:
        img = load_image(p)
        if img is None:
            print("{} is not an image file, skipped".format(p.split("/")[-1]))
            continue
        new_image_paths.append(p)
        new_images.append(img)
    print("Identify {} images in the directory".format(len(new_image_paths)))
    return new_image_paths, new_images


class Faces(object):
    def __init__(
        self,
        image_paths,
        loaded_images,
        aligner,
        verbose=1,
        eval_local=False,
        preprocessing=True,
        no_align=False,
    ):
        self.image_paths = image_paths
        self.verbose = verbose
        self.no_align = no_align
        self.aligner = aligner
        self.margin = 30
        self.org_faces = []
        self.cropped_faces = []
        self.cropped_faces_shape = []
        self.cropped_index = []
        self.start_end_ls = []
        self.callback_idx = []
        self.images_without_face = []
        for i in range(0, len(loaded_images)):
            cur_img = loaded_images[i]
            p = image_paths[i]
            self.org_faces.append(cur_img)

            if not no_align:
                align_img = align(cur_img, self.aligner)
                if align_img is None:
                    print("Find 0 face(s) in {}".format(p.split("/")[-1]))
                    self.images_without_face.append(i)
                    continue

                cur_faces = align_img[0]
            else:
                cur_faces = [cur_img]

            cur_faces = [
                face for face in cur_faces if face.shape[0] != 0 and face.shape[1] != 0
            ]
            cur_shapes = [f.shape[:-1] for f in cur_faces]

            cur_faces_square = []
            if verbose and not no_align:
                print("Find {} face(s) in {}".format(len(cur_faces), p.split("/")[-1]))
            if eval_local:
                cur_faces = cur_faces[:1]

            for img in cur_faces:
                if eval_local:
                    base = resize(img, (IMG_SIZE, IMG_SIZE))
                else:
                    long_size = max([img.shape[1], img.shape[0]]) + self.margin

                    base = np.ones((long_size, long_size, 3)) * np.mean(
                        img, axis=(0, 1)
                    )

                    start1, end1 = get_ends(long_size, img.shape[0])
                    start2, end2 = get_ends(long_size, img.shape[1])

                    base[start1:end1, start2:end2, :] = img
                    cur_start_end = (start1, end1, start2, end2)
                    self.start_end_ls.append(cur_start_end)

                cur_faces_square.append(base)
            cur_faces_square = [
                resize(f, (IMG_SIZE, IMG_SIZE)) for f in cur_faces_square
            ]
            self.cropped_faces.extend(cur_faces_square)

            if not self.no_align:
                cur_index = align_img[1]
                self.cropped_faces_shape.extend(cur_shapes)
                self.cropped_index.extend(cur_index[: len(cur_faces_square)])
                self.callback_idx.extend([i] * len(cur_faces_square))

        if len(self.cropped_faces) == 0:
            return

        self.cropped_faces = np.array(self.cropped_faces)

        if preprocessing:
            self.cropped_faces = preprocess(self.cropped_faces, PREPROCESS)

        self.cloaked_faces = np.copy(self.org_faces)

    def merge_faces(self, protected_images, original_images):
        if self.no_align:
            return np.clip(protected_images, 0.0, 255.0), self.images_without_face

        self.cloaked_faces = np.copy(self.org_faces)

        for i in range(len(self.cropped_faces)):
            cur_protected = protected_images[i]
            cur_original = original_images[i]

            org_shape = self.cropped_faces_shape[i]

            old_square_shape = max([org_shape[0], org_shape[1]]) + self.margin

            cur_protected = resize(cur_protected, (old_square_shape, old_square_shape))
            cur_original = resize(cur_original, (old_square_shape, old_square_shape))

            start1, end1, start2, end2 = self.start_end_ls[i]

            reshape_cloak = cur_protected - cur_original
            reshape_cloak = reshape_cloak[start1:end1, start2:end2, :]

            callback_id = self.callback_idx[i]
            bb = self.cropped_index[i]
            self.cloaked_faces[callback_id][bb[0] : bb[2], bb[1] : bb[3], :] += (
                reshape_cloak
            )

        for i in range(0, len(self.cloaked_faces)):
            self.cloaked_faces[i] = np.clip(self.cloaked_faces[i], 0.0, 255.0)
        return self.cloaked_faces, self.images_without_face


def get_ends(longsize, window):
    start = (longsize - window) // 2
    end = start + window
    return start, end


def resize(img, sz):
    assert np.min(img) >= 0 and np.max(img) <= 255.0
    im_data = utils.array_to_img(img).resize((sz[1], sz[0]))
    im_data = utils.img_to_array(im_data)
    return im_data


def preprocess(X, method):
    assert method in {"raw", "imagenet", "inception", "mnist"}

    if method == "raw":
        pass
    elif method == "imagenet":
        X = imagenet_preprocessing(X)
    else:
        raise Exception("unknown method %s" % method)

    return X


def reverse_preprocess(X, method):
    assert method in {"raw", "imagenet", "inception", "mnist"}

    if method == "raw":
        pass
    elif method == "imagenet":
        X = imagenet_reverse_preprocessing(X)
    else:
        raise Exception("unknown method %s" % method)

    return X


def imagenet_preprocessing(x, data_format=None):
    if data_format is None:
        data_format = backend.image_data_format()
    assert data_format in ("channels_last", "channels_first")

    x = np.array(x)
    if data_format == "channels_first":
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]

    mean = [103.939, 116.779, 123.68]
    std = None

    # Zero-center by mean pixel
    if data_format == "channels_first":
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]

    return x


def imagenet_reverse_preprocessing(x, data_format=None):
    x = np.array(x)
    if data_format is None:
        data_format = backend.image_data_format()
    assert data_format in ("channels_last", "channels_first")

    if data_format == "channels_first":
        if x.ndim == 3:
            # Zero-center by mean pixel
            x[0, :, :] += 103.939
            x[1, :, :] += 116.779
            x[2, :, :] += 123.68
            # 'BGR'->'RGB'
            x = x[::-1, :, :]
        else:
            x[:, 0, :, :] += 103.939
            x[:, 1, :, :] += 116.779
            x[:, 2, :, :] += 123.68
            x = x[:, ::-1, :, :]
    else:
        # Zero-center by mean pixel
        x[..., 0] += 103.939
        x[..., 1] += 116.779
        x[..., 2] += 123.68
        # 'BGR'->'RGB'
        x = x[..., ::-1]
    return x


def reverse_process_cloaked(x, preprocess="imagenet"):
    return reverse_preprocess(x, preprocess)


def load_extractor(name):
    hash_map = {
        "extractor_2": "ce703d481db2b83513bbdafa27434703",
        "extractor_0": "94854151fd9077997d69ceda107f9c6b",
    }
    assert name in ["extractor_2", "extractor_0"]
    model_file = utils.get_file(
        fname="{}.h5".format(name),
        origin="http://mirror.cs.uchicago.edu/fawkes/files/{}.h5".format(name),
        md5_hash=hash_map[name],
        cache_subdir="model",
    )

    model = models.load_model(model_file)
    model = Extractor(model)
    return model


class Extractor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, imgs):
        imgs = imgs / 255.0
        embeds = l2_norm(self.model(imgs))
        return embeds

    def __call__(self, x):
        return self.predict(x)


def dump_image(x, filename, format="png", scale=False):
    img = utils.array_to_img(x, scale=scale)
    img.save(filename, format)
    return


def load_embeddings(feature_extractors_names):
    model_dir = os.path.join(os.path.expanduser("~"), ".fawkes")
    for extractor_name in feature_extractors_names:
        fp = gzip.open(
            os.path.join(model_dir, "{}_emb.p.gz".format(extractor_name)), "rb"
        )
        path2emb = pickle.load(fp)
        fp.close()

    return path2emb


def extractor_ls_predict(feature_extractors_ls, X):
    feature_ls = []
    for extractor in feature_extractors_ls:
        cur_features = extractor.predict(X)
        feature_ls.append(cur_features)
    concated_feature_ls = np.concatenate(feature_ls, axis=1)
    return concated_feature_ls


def pairwise_l2_distance(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)
    SqA = A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = tf.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output
