#!/usr/bin/env python
""" Explore different preprocessing transformation on data before training """
import cv2
from selam.utils import img
from keras.preprocessing.image import ImageDataGenerator


def createDataGenerator():
    datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    return datagen


def main():
    path = './examples/dataset/robosub16/FRONT/0-264_buoys'
    imgs = img.get_jpgs(path)
    # Define data augmentation scheme
    # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    chosen_img = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
    data = chosen_img.reshape((1, ) + chosen_img.shape)
    datagen = createDataGenerator()

    for i, batch in enumerate(datagen.flow(data, batch_size=1,
                              save_to_dir='examples/output/augmented', save_prefix='buoy',
                              save_format='jpeg')):
        # Return only 30 augmented images
        if i > 30:
            break


if __name__ == '__main__':
    main()
