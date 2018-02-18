import time
import random
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc

def crop(image, top_percent, bottom_percent):
    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    return image[top:bottom, :]

def flip_image(image, steering_angle):
    return np.fliplr(image), -1 * steering_angle

def random_gamma(image):
    #https://stackoverflow.com/a/41061351/3220303
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def random_shear(image, steering_angle, shear_range=200):
    #https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle

def generate_new_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9):
    flip = random.choice([0, 1]) 
    if flip == 1:
        image, steering_angle = random_shear(image, steering_angle)
    
    flip = random.choice([0, 1]) 
    if flip == 1:
        image, steering_angle = flip_image(image, steering_angle)

    image = crop(image, top_crop_percent, bottom_crop_percent)
    image = random_gamma(image)
    image = scipy.misc.imresize(image, resize_dim)

    return image, steering_angle

def get_next_image_files(batch_size=64):
    data = pd.read_csv('./data/driving_log.csv')
    num_of_img = len(data)
    rnd_indices = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in rnd_indices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + 0.229
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - 0.229
            image_files_and_angles.append((img, angle))

    return image_files_and_angles

def generate_next_batch(batch_size=64):
    while True:
        X_batch = []
        y_batch = []
        images = get_next_image_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread('./data/' + img_file)
            raw_angle = angle
            new_image, new_angle = generate_new_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)
        yield np.array(X_batch), np.array(y_batch)

# save model (.json) and weights (.h5)
# keras does not overwrite existing model so must generate unique filename
def save_model(model):
    timestamp = str(int(time.time()))
    name = 'model' + timestamp
    json_string = model.to_json()
    with open(name + '.json', 'w') as out:
        json.dump(json_string, out)
    model.save_weights(name + '.h5')
