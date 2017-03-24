import math
import numpy as np
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import os
import random
'''
def flatten_matrix_with_first_dim_kept(mat):
    mat = np.array(mat)
    data_size = 1
    #data_size = reduce(lambda data_size, y: data_size*y, mat.shape[1:])
    #new_shape = tuple([len(mat), data_size])

    #mat = mat.reshape(new_shape)
    mat = mat.reshape(-1, 28,28,1)

    return mat
'''

def normalize_image(img):
    return Image.fromarray(np.array(img) / 255)


def prepare_image(image,
                  scale_image_size=(-1, -1),
                  normalize_values=False,
                  change_to_gray=False,
                  smooth_factor=1):
    new_img = image

    if not scale_image_size == (-1, -1):
        new_img = new_img.resize(scale_image_size)

    if change_to_gray:
        new_img = new_img.convert('L')

    if not smooth_factor == 1:
        new_img = new_img.filter(ImageFilter.GaussianBlur(smooth_factor))

    if normalize_values:
        new_img = normalize_image(new_img)

    return new_img


def prepare_dataset(images_path,
                    scale_image_size=(-1, -1),
                    normalize_values=False,
                    change_to_gray=False,
                    smooth_factor=1,
                    in_memory=False,
                    target_path='/home/osboxes/PycharmProjects/ML_final_project/workspace/tmp_images/'):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for file_name in os.listdir(images_path):
        curr_path = images_path + '/' + file_name

        # If file
        if os.path.isfile(curr_path):
            img = load_image(curr_path, manipulateImage=False)
            img = prepare_image(image=img,
                                scale_image_size=scale_image_size,
                                normalize_values=normalize_values,
                                change_to_gray=change_to_gray,
                                smooth_factor=smooth_factor)
            img.save(target_path + '/' + file_name)
        # If dir
        elif os.path.isdir(curr_path):
            # Checks if path exists in workspace
            if not os.path.exists(target_path + '/' + file_name):
                os.makedirs(target_path + '/' + file_name)

            for img_name in os.listdir(curr_path):
                img_path = curr_path + '/' + img_name
                img = load_image(img_path, manipulateImage=False)
                img = prepare_image(image=img,
                                    scale_image_size=scale_image_size,
                                    normalize_values=normalize_values,
                                    change_to_gray=change_to_gray,
                                    smooth_factor=smooth_factor)
                img.save(target_path + '/' + file_name + '/' + img_name)


def load_image(imagepath, manipulateImage=False,randrange=1.0):
    im = Image.open(imagepath)

    # Manipulates image: rotation, contrast, brightness
    if manipulateImage:
        im = manipulate_image(im)

    return im

def manipulate_image(img, randrange=1.0, background_image=None):
    if background_image != None:
        img = add_square_pages(img, background_img=background_image, max_percent_to_crop=0.4)

    randnum = random.uniform(-1 * (5 * randrange), (5 * randrange))
    img = img.rotate(randnum, resample=Image.BICUBIC, expand=False)
    contr = ImageEnhance.Contrast(img)
    randnum = random.uniform(1 - (randrange / 2), 1 + (randrange / 2))
    img = contr.enhance(randnum)
    bright = ImageEnhance.Brightness(img)
    randnum = random.uniform(1 - (randrange / 2), 1 + (randrange / 2))
    img = bright.enhance(randnum)

    return img


def expand_dataset(images_path, scale_factor, add_square_page_background_factor=0, square_page_path=''):
    if add_square_page_background_factor > 0:
        square_background_img = Image.open(square_page_path).convert('L')


    for dir in os.listdir(images_path):
        for image_name in os.listdir(images_path + '/' + dir):
            img = load_image(images_path + '/' + dir + '/' + image_name, manipulateImage=False)
            name = image_name.split('.')

            for i in xrange(scale_factor):
                new_img = manipulate_image(img)
                new_img.save(images_path + '/' + dir + '/' + name[0] + 'v' + str(i) + '.' + name[-1])
            for i in xrange(scale_factor, scale_factor + add_square_page_background_factor):
                new_img = manipulate_image(img, background_image=square_background_img)
                new_img.save(images_path + '/' + dir + '/' + name[0] + 'v' + str(i) + '.' + name[-1])


def add_square_pages(img, background_img, max_percent_to_crop = 0.4):
    percent_1 = random.uniform(0, max_percent_to_crop)
    percent_2 = random.uniform(0, max_percent_to_crop)
    percent_3 = random.uniform(0, max_percent_to_crop)
    percent_4 = random.uniform(0, max_percent_to_crop)

    background_size = background_img.size

    crop_from_top = int(math.floor(background_size[0] * percent_1))
    crop_from_bottom = int(math.floor(background_size[0] * percent_2))
    crop_from_right = int(math.floor(background_size[1] * percent_3))
    crop_from_left = int(math.floor(background_size[1] * percent_4))

    background_tmp = np.array(background_img)
    background_tmp = background_tmp[crop_from_top:background_size[0] - crop_from_bottom,
                     crop_from_left:background_size[1] - crop_from_right]

    background = Image.fromarray(background_tmp)
    background = background.resize(img.size)

    new_img = blend_multiply(img, background)
    return new_img


def blend_multiply(img1,img2):
    img1 = np.array(img1, dtype='float')
    img2 = np.array(img2, dtype='float')
    new_img = np.empty((img1.shape), dtype='float')

    for i in xrange(img1.shape[0]):
        for j in xrange(img1.shape[1]):
            new_img[i, j] = (img1[i, j] * img2[i, j]) / 256

    return Image.fromarray(np.floor(new_img).astype(dtype='uint8'))