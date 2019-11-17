import scipy.misc
import random
import numpy as np
import copy
import sys


img_path = '/home/speit_ie02/Data/KITTI/kitti_data/training/image_2/'
label_path = '/home/speit_ie02/Data/KITTI/kitti_data/training/label_2/'
# FFNet needs a 2D object detection model or 2D object detection result for KITTI evaluation
# define your 2D object detection result here
result_2d_path = ''

# 0: test / 1: train / 2: write test results
toTrain = int(sys.argv[1])

# num_train_img: total number of images for train and val (not only for training)
num_train_img = 7481
num_val_img = 1481

num_train_data = 0
num_val_data = 0

class_dict = {
    'Car': 0,
    'Van': 0,
    'Truck': 3,
    'Pedestrian': 1,
    'Person_sitting': 3,
    'Cyclist': 2,
    'Tram': 3,
    'Misc': 3,
    'DontCare': 3
}
target_element = [0, 3, 4, 5, 6, 7, 8, 9, 10]

train_batch_pointer = 0
PI = 3.141592654

angle_diff_max = 30


def generate_one_hot_list(index, depth):
    one_hot_list = []
    for i in range(depth):
        one_hot_list.append(0)
    one_hot_list[index] = 1
    return one_hot_list

def determine_angle(sin_value, cos_value):
    if cos_value >= 0:
        return np.arcsin(sin_value)
    elif sin_value >= 0:
        return PI - np.arcsin(sin_value)
    elif sin_value < 0:
        return -PI - np.arcsin(sin_value)

# no classification in the multi-bin mechanism
def determine_average_degree(angle_array, to_display):
    a_value = []
    a_diff_value = []
    a_sin = []
    a_cos = []
    for angle_index in range(len(angle_array)):
        sin_value = angle_array[angle_index][0][0]
        cos_value = angle_array[angle_index][0][1]
        sc_set = np.array([sin_value, cos_value])
        l2_normed_sc_set = sc_set / np.sqrt(np.sum(sc_set ** 2))
        sin_value = l2_normed_sc_set[0]
        cos_value = l2_normed_sc_set[1]
        angle_value = determine_angle(sin_value, cos_value)
        a_diff_value.append(angle_value*180/PI)
        angle_value = angle_value - PI/4 + PI*angle_index/2
        sin_value = np.sin(angle_value); a_sin.append(sin_value)
        cos_value = np.cos(angle_value); a_cos.append(cos_value)
        angle_value = angle_value*180/PI
        angle_value = angle_value % 360
        a_value.append(angle_value)

    # error excluding algorithm
    for i in range(len(a_value)):
        a_rest = []
        for j in range(len(a_value)):
            if j != i:
                a_rest.append(a_value[j])
        x1 = abs(a_value[i] - a_rest[0])
        x2 = abs(a_value[i] - a_rest[1])
        x3 = abs(a_value[i] - a_rest[2])
        delta_angle = max(x1, x2, x3) - min(x1, x2, x3)
        if x1 > delta_angle and x2 > delta_angle and x3 > delta_angle:
            if  delta_angle < 2*angle_diff_max:
                del a_sin[i]
                del a_cos[i]
                break

    variance_degree = np.var(a_value)

    sin_value = sum(a_sin) / len(a_sin)
    cos_value = sum(a_cos) / len(a_cos)
    angle_value = determine_angle(sin_value, cos_value)
    if to_display:
        print(a_value)
        # print(a_diff_value)
    return angle_value*180/PI, variance_degree

def find_second_largest_element(a):
    index = np.argmax(a)
    a[index] = 0
    index = np.argmax(a)
    return a[index]

# each line has 16 elements
# 0: object class
# 1: if truncated
# 2: if blocked
# 3: observed angle
# 4~7: 2D bounding box: xmin, ymin, xmax, ymax
# 8~10: 3D bounding box dimensions: height, width, length
# 11~13: 3D bounding box location: x, y, z
# 14: yaw angle
# 15: detection confidence
def get_txt_data(filename):
    with open(filename, 'r') as f:
        # get object class and characteristics
        object_data = []
        for ln in f.readlines():
            line_data = ln.strip().split(" ")[:]
            line_target_data = []
            if line_data[0] == 'Pedestrian':
                for i in target_element:
                    line_target_data.append(line_data[i])
                object_data.append(line_target_data)
        return object_data

def organize_train_data():
    img_epoch = []
    label_epoch = []
    for i in range(num_train_img):
        if i < 2000 or i > 3480:
            seqname = str(i).zfill(6)
            img = scipy.misc.imread(img_path + seqname + '.png')
            label = get_txt_data(label_path + seqname + '.txt')
            for j in range(len(label)):
                target_img = copy.deepcopy(img[int(float(label[j][3])):int(float(label[j][5]))+1,int(float(label[j][2])):int(float(label[j][4]))+1])
                target_img = target_img.astype(np.float32) / 255.0
                target_img = scipy.misc.imresize(target_img, [224, 224])
                target_label = copy.deepcopy(label[j])
                img_epoch.append(target_img)
                label_epoch.append(target_label)
            print('train data', i, 'complete')
    dataset = list(zip(img_epoch, label_epoch))
    random.shuffle(dataset)
    img_epoch, label_epoch = zip(*dataset)
    return img_epoch, label_epoch

def organize_val_data():
    img_epoch = []
    label_epoch = []
    for i in range(num_val_img):
        seqname = str(2000+i).zfill(6)
        img = scipy.misc.imread(img_path + seqname + '.png')
        label = get_txt_data(label_path + seqname + '.txt')
        for j in range(len(label)):
            target_img = copy.deepcopy(img[int(float(label[j][3])):int(float(label[j][5]))+1,int(float(label[j][2])):int(float(label[j][4]))+1])
            target_img = target_img.astype(np.float32) / 255.0
            target_img = scipy.misc.imresize(target_img, [224, 224])
            target_label = copy.deepcopy(label[j])
            img_epoch.append(target_img)
            label_epoch.append(target_label)
        print('val data', i+2000, 'complete')
    dataset = list(zip(img_epoch, label_epoch))
    random.shuffle(dataset)
    img_epoch, label_epoch = zip(*dataset)
    return img_epoch, label_epoch


img_epoch_train = []
label_epoch_train = []
img_epoch_val = []
boxsize_epoch_val = []
d_epoch_val = []
c_epoch_val = []
a_epoch_val = []
a = []

if toTrain == 1:
    img_epoch_train, label_epoch_train = organize_train_data()
    num_train_data = len(label_epoch_train)

if toTrain == 0 or toTrain == 1:

    img_epoch_val, label_epoch_val = organize_val_data()
    num_val_data = len(label_epoch_val)

    box_min_epoch_val = np.array([label[2:4] for label in label_epoch_val]).astype(np.float32)
    box_max_epoch_val = np.array([label[4:6] for label in label_epoch_val]).astype(np.float32)
    boxsize_epoch_val = box_max_epoch_val - box_min_epoch_val

    for label in label_epoch_val:
        c_epoch_val.append(generate_one_hot_list(class_dict[label[0]], 2))
        d_epoch_val.append(label[6:9])
        a_epoch_val.append([label[1]])

    a = np.zeros(num_val_data)
    for i in range(num_val_data):
        a[i] += i


def get_train_data(batch_size):
    global train_batch_pointer
    img_batch = []
    label_batch = []
    for i in range(batch_size):
        img_batch.append(img_epoch_train[train_batch_pointer])
        label_batch.append(label_epoch_train[train_batch_pointer])
        train_batch_pointer += 1
        if train_batch_pointer >= num_train_data:
            train_batch_pointer = 0

    box_batch = [label[2:6] for label in label_batch]
    box_min_batch = np.array([label[2:4] for label in label_batch]).astype(np.float32)
    box_max_batch = np.array([label[4:6] for label in label_batch]).astype(np.float32)
    boxsize_batch = box_max_batch - box_min_batch
    d_batch = [label[6:9] for label in label_batch]
    c_batch = [generate_one_hot_list(class_dict[label[0]], 2) for label in label_batch]
    a_batch = [[label[1]] for label in label_batch]
    return img_batch, boxsize_batch, d_batch, c_batch, a_batch

def extract_random_val_batch(batch_size):
    random.shuffle(a)
    img_batch = []
    boxsize_batch = []
    d_batch = []
    c_batch = []
    a_batch = []
    for i in range(batch_size):
        sample_index = int(a[i])
        img_batch.append(img_epoch_val[sample_index])
        boxsize_batch.append(boxsize_epoch_val[sample_index])
        d_batch.append(d_epoch_val[sample_index])
        c_batch.append(c_epoch_val[sample_index])
        a_batch.append(a_epoch_val[sample_index])
    return img_batch, boxsize_batch, d_batch, c_batch, a_batch

