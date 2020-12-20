import csv
import os
from pathlib import Path

import chardet
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image as Image
from sklearn.model_selection import train_test_split

font = cv2.FONT_HERSHEY_SIMPLEX


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


root = get_project_root()

print(root)

class ADataLoader:

    def __init__(self, X_data_path, Y_data_path, task):
        self.X_data_path = X_data_path
        self.Y_data_path = Y_data_path

        self.task = task

        random.seed(42)

        self.X_train, self.Y_train, self.X_test, self.Y_test, = self.load_data()

    def load_data(self):
        # Load the labels data
        with open(self.Y_data_path, 'rb') as f:
            result = chardet.detect(f.read())
            Y_df = pd.read_csv(self.Y_data_path,
                               encoding=result['encoding'],
                               error_bad_lines=False,
                               quoting=csv.QUOTE_NONE,
                               delimiter='\t',
                               index_col=[0],
                               engine='python')

        Y_df.columns = Y_df.columns.str.strip()
        Y_df.fillna(0)

        # print number of images in each dataset
        print('There are %d total images.' % len(Y_df))

        # Extract and crop faces
        X, Y = self.detect_scale(Y_df)

        # print number of images in each dataset
        print('There are %d images after feature selection.' % len(Y))

        return self.split_data(X, Y)

    @staticmethod
    def split_data(X, Y):
        # pre-process the data for Keras
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=True)

        print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

        return X_train, Y_train, X_test, Y_test

    def detect_scale(self, Y_df):
        face_cascade = cv2.CascadeClassifier(
            os.path.join(str(root), 'helpers/haarcascade/haarcascade_frontalface_default.xml'))
        eye_cascade = cv2.CascadeClassifier(
            os.path.join(str(root), 'helpers/haarcascade/haarcascade_eye.xml'))
        list_image_tensors = []
        list_image_labels = []

        for filename in Y_df['img_name']:
            # load color (RGB) image
            img = cv2.imread(self.X_data_path + filename, 1)
            # convert RGB image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # detect the faces in the image
            faces = face_cascade.detectMultiScale(gray, 1.3, 7)

            if faces is not ():
                # get bounding box if face detected

                (x, y, w, h) = faces[0]
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) >= 2:
                    # loads RGB image as PIL.Image.Image type
                    img = cv2.resize(roi_gray, (178, 218), interpolation=cv2.INTER_CUBIC)

                    # convert PIL.Image.Image type to 3D tensor with shape (128, 128, 1)
                    img = Image.img_to_array(img)

                    list_image_tensors.append(img)

                    if Y_df[self.task][int(filename.split('.')[0])] == 1:
                        list_image_labels.append(1)
                    else:
                        list_image_labels.append(0)

                    # # convert BGR image to RGB for plotting
                    # cv_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
                    # plt.imshow(cv_rgb)
                    # plt.show()

        X = np.array(list_image_tensors, dtype="float") / 255.0
        Y = np.array(list_image_labels)

        return X, Y

    def get_train_data(self):
        return self.X_train, self.Y_train

    def get_test_data(self):
        return self.X_test, self.Y_test
