#!/bin/bash
import os
import sys
import random
import cv2
import numpy as np
import xgboost as xgb

from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from keras.preprocessing.image import ImageDataGenerator

from selam.utils import img


def sample_negative(img, rect,  n=1, size=(100, 100)):
    """ Sample n negative samples randomly
        @param rect: [x1, y1, x2, y2]
        @param n: number of negative samples
        @param size: size of negative window
    """
    samples = []
    maxHeight, maxWidth = img.shape[:-1]
    width = abs(rect[0] - rect[2])
    height = abs(rect[1] - rect[3])
    while len(samples) != n:
        tmpX = int(random.random() * (maxWidth - width))
        tmpY = int(random.random() * (maxHeight - height))
        isNotOverlapX = tmpX + width < rect[0]  or tmpX > rect[2]
        isNotOverlapY = tmpY + height < rect[1]  or tmpY > rect[3]
        # Only accepts sample that does not overlap with ground truth
        if isNotOverlapX and isNotOverlapY:
            samples.append(cv2.resize(
                img[tmpY: tmpY + height, tmpX: tmpX + width], size))
    return samples

def get_roi(img, rect, size=(100, 100)):
    """ Return extracted bounding box given 4 corners of a rectangle
        size: size of training image
        @return roi, [x1, y1, x2, y2]
    """
    xpos = rect[0::2]
    ypos = rect[1::2]
    y = [int(min(ypos)), int(max(ypos))]
    x = [int(min(xpos)), int(max(xpos))]
    roi = img[y[0]:y[1], x[0]:x[1]]
    return cv2.resize(roi, size), [x[0], y[0], x[1], y[1]]


def get_jpgs(dirpath, skip=0, resize=None):
    """ Returns all images located in given dirpath
        skip : number of frames skip to reduce computation time
        resize: scale factor for resize

    """
    filenames = os.listdir(dirpath)
    # Only attempt to parse and sort files that end with .jpg
    filenames = [filename for filename in filenames
                 if filename.endswith(".jpg") or filename.endswith(".png")]
    filenames.sort(key=lambda x: int(x.split('.', 1)[0]))
    frames = [cv2.imread('{}/{}'.format(dirpath, filename))
              for filename in filenames]
    out = frames[0::skip] if skip > 0 else frames
    print('Read {} images from {}'.format(len(out), dirpath))
    if resize:
        new_size = (out[0].shape[1] / resize, out[0].shape[0] / resize)
        return map(lambda x: cv2.resize(x, new_size), out)
    return out


def extract_training(dataset_path, annotation):
    """ Returns a list of labelled images as positive training data
        Uses default size of 100 x 100 as training patch
        @return positive samples, negative samples
    """
    positives = []
    negatives = []
    imgs = get_jpgs(dataset_path)
    with open(annotation) as ann:
        for i, label in zip(imgs, ann):
            rect = map(float, label.rstrip().split(','))
            if rect[0] > 0:
                roi, coord = get_roi(i, rect)
                negatives.extend(sample_negative(i, coord))
                positives.append(roi)
    print("{} positive samples".format(len(positives)))
    print("{} negative samples".format(len(negatives)))
    return positives, negatives


def augment_data(imgs, augment_dir, prefix, n=20):
    """ Augment imgs with various transformations 
        @param augment_dir: directory to save augmented images
        @param prefix: prefix of filename
        @param n: number of transformations per image
    """
    n_samples = len(imgs)
    datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    for i in imgs:
        selected = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        selected = selected.reshape((1, ) + selected.shape)
        for x, batch in enumerate(datagen.flow(selected, batch_size=1,
                                  save_to_dir=augment_dir,
                                  save_prefix=prefix,
                                  save_format='jpeg')):
            if x > n:
                break


def kfold(x, y, eval_size=0.10):
    """ Split dataset into training set and validation set
        @param eval_size: percentage of data used for evaluation
        @return X_train, X_valid, Y_train, Y_valid
    """
    return train_test_split(x, y, test_size=eval_size, random_state=0)


def std_zscore(X):
    """ Z-score standardization by subtracting mean and divided by standard
    deviation of dataset
    """
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)


def std_minmax(X):
    scaler = preprocessing.MinMaxScaler().fit(X)
    return scaler.transform(X)


def reduce_pca(X, h, w, n=15, display=True):
    """ Performs PCA decomposition using n components """
    pca = PCA(n_components=n, svd_solver='randomized',
              whiten=True).fit(X)
    eigenfaces = pca.components_.reshape((n, h, w, -1))
    if display:
        for i in eigenfaces:
            cv2.imshow('PC', np.uint8(img.normUnity(np.mean(i, axis=2)) * 255))
            cv2.waitKey(0)
    return pca.transform(X)


def reduce_nmf(X, h, w, n=15, display=False):
    """ Performs Non-negative matrix factorization using n components """
    model = NMF(n_components=n, init='random', random_state=0).fit(X)
    components = model.components_.reshape((n, h, w, -1))
    if display:
        for i in components:
            cv2.imshow('PC', np.uint8(img.normUnity(np.mean(i, axis=2)) * 255))
            cv2.waitKey(0)
    return model.transform(X)


def classify_svm(X_train, Y_train):
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf.fit(X_train, Y_train)
    return clf


def classify_rf(X_train, Y_train):
    param_grid = {'n_estimators': [50, 200, 700],
                  'max_features': ['auto', 'sqrt', 'log2']}
    clf = GridSearchCV(RandomForestClassifier(n_estimators=500, oob_score=True), param_grid)
    clf.fit(X_train, Y_train)
    return clf


def classify_gp(X, Y):
    # Using same lengthscale for all features
    kernel = 1.0 * RBF([1.0])
    gpc_rbf = GaussianProcessClassifier(kernel=kernel).fit(X, Y)
    return gpc_rbf


def classify_xgb(X, Y):
    xgb_model = xgb.XGBClassifier()
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
		  'objective':['binary:logistic'],
		  'learning_rate': [0.05], #so called `eta` value
		  'max_depth': [6],
		  'min_child_weight': [11],
		  'silent': [1],
		  'subsample': [0.8],
		  'colsample_bytree': [0.7],
		  'n_estimators': [5], #number of trees, change it to 1000 for better results
		  'missing':[-999],
		  'seed': [1337]}
    clf = GridSearchCV(xgb_model, parameters)
    clf.fit(X, Y)
    return clf


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python extract_region.py <dataset directory> <annotation file> <prefix> \n")
        exit()
    positives, negatives = extract_training(sys.argv[1], sys.argv[2])
