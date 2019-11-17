# FFNet main program interface
# command:
# "python main.py 1" for training
# "python main.py 0" for validation
# "python main.py 2" for orientation results writedown (of KITTI 7518 test images)


import tensorflow as tf
import tensorflow.contrib.slim as slim
import data_processing
import numpy as np
import scipy.misc
import os
import sys
import time
import math
import copy
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_batch = 22500
learning_rate = 1e-3
# after num_batch_lv2 batches, learning rate = lr_lv2
num_batch_lv2 = 6000
lr_lv2 = 1e-4
# after num_batch_lv3 batches, learning rate = lr_lv3
num_batch_lv3 = 12000
lr_lv3 = 1e-4

batch_size = 32
num_bin = 4

PI = data_processing.PI
ckpt_dir = './checkpoint_ffnet/'


img = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
boxsize = tf.placeholder(tf.float32, shape = [None, 2])
d_label = tf.placeholder(tf.float32, shape = [None, 3])
c_label = tf.placeholder(tf.float32, shape = [None, 2])
# orientation label, presented as [-pi, pi)
a_label = tf.placeholder(tf.float32, shape = [None, 1])
keep_prob = tf.placeholder(tf.float32, shape=None)


def build_model():

    def LeakyReLU(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def determine_angle_loss(angle_prediction, angle_label, index):
        error_total = 0
        angle_tmp = -PI*index/2
        for i in range(batch_size):
            sin_prediction = angle_prediction[i][0]
            cos_prediction = angle_prediction[i][1]
            sin_label = tf.sin(angle_label[i][0] + PI/4 + angle_tmp)
            cos_label = tf.cos(angle_label[i][0] + PI/4 + angle_tmp)
            error_total += (1-sin_label*sin_prediction-cos_label*cos_prediction)
        return error_total / batch_size

    def determine_dimension_consistency_loss(dimension_prediction, angle_label, boxsize_input):
        error_total = 0
        for i in range(batch_size):
            sin_label = tf.sin(angle_label[i][0])
            cos_label = tf.cos(angle_label[i][0])
            sin_label = tf.abs(sin_label)
            cos_label = tf.abs(cos_label)
            w1_ = dimension_prediction[i][1] * sin_label + dimension_prediction[i][2] * cos_label
            error = dimension_prediction[i][0] * boxsize_input[i][0] - w1_ * boxsize_input[i][1]
            error_total += tf.abs(error)
        return error_total / batch_size

    def determine_angle_consistency_loss(angle_prediction, dimension_label, boxsize_input, index):
        error_total = 0
        angle_offset = -PI*index/2 + PI/4
        sin_offset = tf.sin(angle_offset)
        cos_offset = tf.cos(angle_offset)
        for i in range(batch_size):
            sin_prediction = angle_prediction[i][0]
            cos_prediction = angle_prediction[i][1]
            sin_adjusted = sin_prediction * cos_offset - cos_prediction * sin_offset
            cos_adjusted = cos_prediction * cos_offset + sin_prediction * sin_offset
            w1_ = dimension_label[i][1] * sin_adjusted + dimension_label[i][2] * cos_adjusted
            error = dimension_label[i][0] * boxsize_input[i][0] - w1_ * boxsize_input[i][1]
            error_total += tf.abs(error)
        return error_total / batch_size

    # VGG16 main frame
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(img, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        conv_c = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        conv_c = slim.max_pool2d(conv_c, [2, 2], scope='pool3')
        net = slim.repeat(conv_c, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        conv_img = tf.contrib.layers.flatten(net)
        conv_confidence = tf.contrib.layers.flatten(conv_c)

        net_boxsize = slim.fully_connected(boxsize, 256, activation_fn=None, scope='fc_bs1')
        net_boxsize = LeakyReLU(net_boxsize, 0)
        net_boxsize = slim.fully_connected(net_boxsize, 2048, activation_fn=None, scope='fc_bs2')

        confidence = slim.fully_connected(conv_confidence, 1024, activation_fn=None, scope='fc_c1')
        confidence = LeakyReLU(confidence, 0)
        confidence = slim.dropout(confidence, keep_prob, scope='dropout_c1')
        confidence = slim.fully_connected(confidence, 2, activation_fn=None, scope='fc_c2')
        loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=c_label, logits=confidence))

        c_label_intermediate = c_label
        # c_label_intermediate = confidence
        tf.stop_gradient(c_label_intermediate)
        confidence_feedback = slim.fully_connected(c_label_intermediate, 256, activation_fn=None, scope='fc_cfb1')
        confidence_feedback = LeakyReLU(confidence_feedback, 0.1)
        confidence_feedback = slim.fully_connected(confidence_feedback, 2048, activation_fn=None, scope='fc_cfb2')

        conv_img_concat_d = tf.concat([conv_img, confidence_feedback], 1)

        dimension = slim.fully_connected(conv_img_concat_d, 512, activation_fn=None, scope='fc_d1')
        dimension = LeakyReLU(dimension, 0)
        dimension = slim.dropout(dimension, keep_prob, scope='dropout_d1')
        dimension = slim.fully_connected(dimension, 3, activation_fn=None, scope='fc_d2')
        loss_d = tf.losses.mean_squared_error(d_label, dimension)
        loss_d_consistency = determine_dimension_consistency_loss(dimension, a_label, boxsize)

        d_label_intermediate = d_label
        # d_label_intermediate = dimension
        tf.stop_gradient(d_label_intermediate)
        dimension_feedback = slim.fully_connected(d_label_intermediate, 256, activation_fn=None, scope='fc_dfb1')
        dimension_feedback = LeakyReLU(dimension_feedback, 0)
        dimension_feedback = slim.fully_connected(dimension_feedback, 2048, activation_fn=None, scope='fc_dfb2')

        conv_img_concat_a = tf.concat([conv_img, dimension_feedback], 1)
        conv_img_concat_a = tf.concat([conv_img_concat_a, net_boxsize], 1)

        angle1 = slim.fully_connected(conv_img_concat_a, 512, activation_fn=None, scope='fc_a11')
        angle1 = LeakyReLU(angle1, 0)
        angle1 = slim.dropout(angle1, keep_prob, scope='dropout_a11')
        angle1 = slim.fully_connected(angle1, 2, activation_fn=None, scope='fc_a12')
        angle1 = tf.nn.l2_normalize(angle1, axis=1)
        loss_a1 = determine_angle_loss(angle1, a_label, 0)
        loss_a1_consistency = determine_angle_consistency_loss(angle1, d_label, boxsize, 0)

        angle2 = slim.fully_connected(conv_img_concat_a, 512, activation_fn=None, scope='fc_a21')
        angle2 = LeakyReLU(angle2, 0)
        angle2 = slim.dropout(angle2, keep_prob, scope='dropout_a21')
        angle2 = slim.fully_connected(angle2, 2, activation_fn=None, scope='fc_a22')
        angle2 = tf.nn.l2_normalize(angle2, axis=1)
        loss_a2 = determine_angle_loss(angle2, a_label, 1)
        loss_a2_consistency = determine_angle_consistency_loss(angle2, d_label, boxsize, 1)

        angle3 = slim.fully_connected(conv_img_concat_a, 512, activation_fn=None, scope='fc_a31')
        angle3 = LeakyReLU(angle3, 0)
        angle3 = slim.dropout(angle3, keep_prob, scope='dropout_a31')
        angle3 = slim.fully_connected(angle3, 2, activation_fn=None, scope='fc_a32')
        angle3 = tf.nn.l2_normalize(angle3, axis=1)
        loss_a3 = determine_angle_loss(angle3, a_label, 2)
        loss_a3_consistency = determine_angle_consistency_loss(angle3, d_label, boxsize, 2)

        angle4 = slim.fully_connected(conv_img_concat_a, 512, activation_fn=None, scope='fc_a41')
        angle4 = LeakyReLU(angle4, 0)
        angle4 = slim.dropout(angle4, keep_prob, scope='dropout_a41')
        angle4 = slim.fully_connected(angle4, 2, activation_fn=None, scope='fc_a42')
        angle4 = tf.nn.l2_normalize(angle4, axis=1)
        loss_a4 = determine_angle_loss(angle4, a_label, 3)
        loss_a4_consistency = determine_angle_consistency_loss(angle4, d_label, boxsize, 3)

        angle_set = [angle1, angle2, angle3, angle4]
        loss_a = loss_a1 + loss_a2 + loss_a3 + loss_a4
        loss_a_consistency = loss_a1_consistency + loss_a2_consistency + loss_a3_consistency + loss_a4_consistency

        # total_loss = loss_c + loss_d + loss_a + 0.01 * loss_d_consistency + 0.01 * loss_a_consistency
        total_loss = loss_c + loss_d + loss_a

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

        return dimension, angle_set, total_loss, optimizer, loss_d, loss_c, loss_a, loss_d_consistency, loss_a_consistency


def train():
    print(); print("training:"); print()
    print("Number of training data: %d" % data_processing.num_train_data)
    print("Number of validation data: %d" % data_processing.num_val_data)

    result_dir = './training_process/'
    if os.path.isdir(result_dir) == True:
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    seq_file = open('training_process/seq.txt', 'w')
    loss_d_file = open('training_process/loss_d.txt', 'w')
    loss_d_consistency_file = open('training_process/loss_d_consistency.txt', 'w')
    loss_c_file = open('training_process/loss_c.txt', 'w')
    loss_a_file = open('training_process/loss_a.txt', 'w')
    loss_a_consistency_file = open('training_process/loss_a_consistency.txt', 'w')
    loss_file = open('training_process/loss.txt', 'w')

    dimension, angle_set, loss, optimizer, loss_d, loss_c, loss_a, loss_d_consistency, loss_a_consistency = build_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if not ckpt:
        print(); print("checkpoint not found."); print()
    sess.run(tf.global_variables_initializer())
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for batch in range(num_batch):

        if batch >= num_batch_lv2:
            learning_rate = lr_lv2
        if batch >= num_batch_lv3:
            learning_rate = lr_lv3

        img_batch, boxsize_batch, d_batch, c_batch, a_batch = data_processing.get_train_data(batch_size)

        _, loss_output, dimension_output = \
            sess.run([optimizer, loss, dimension],
                    feed_dict={img: img_batch,
                               boxsize: boxsize_batch,
                               d_label: d_batch,
                               c_label: c_batch,
                               a_label: a_batch,
                               keep_prob: 0.5})

        if math.isnan(loss_output):
            sys.exit(0)

        if (batch + 1) % 20 == 0:
            img_val_batch, boxsize_val_batch, d_val_batch, c_val_batch, a_val_batch = \
                data_processing.extract_random_val_batch(batch_size)

            loss_d_value_val, loss_c_value_val, loss_a_value_val, loss_d_consistency_value_val, loss_a_consistency_value_val = \
                sess.run([loss_d, loss_c, loss_a, loss_d_consistency, loss_a_consistency],
                        feed_dict={img: img_val_batch,
                                   boxsize: boxsize_val_batch,
                                   d_label: d_val_batch,
                                   c_label: c_val_batch,
                                   a_label: a_val_batch,
                                   keep_prob: 1.0})
            loss_value_train, loss_d_value_train, loss_c_value_train, loss_a_value_train, loss_d_consistency_value_train, loss_a_consistency_value_train = \
                sess.run([loss, loss_d, loss_c, loss_a, loss_d_consistency, loss_a_consistency],
                        feed_dict={img: img_batch,
                                   boxsize: boxsize_batch,
                                   d_label: d_batch,
                                   c_label: c_batch,
                                   a_label: a_batch,
                                   keep_prob: 1.0})

            print("Batch:", batch + 1)
            print("VAL BATCH: loss_d: %g, loss_c: %g, loss_a: %g, loss_d_consistency: %g, loss_a_consistency: %g"\
                  % (loss_d_value_val, loss_c_value_val, loss_a_value_val, loss_d_consistency_value_val, loss_a_consistency_value_val))
            print("TRAIN BATCH: loss_d: %g, loss_c: %g, loss_a: %g, loss_d_consistency: %g, loss_a_consistency: %g"\
                  % (loss_d_value_train, loss_c_value_train, loss_a_value_train, loss_d_consistency_value_train, loss_a_consistency_value_train))

            loss_d_file.write(str(loss_d_value_train) + '\n')
            loss_d_consistency_file.write(str(loss_d_consistency_value_train) + '\n')
            loss_a_file.write(str(loss_a_value_train) + '\n')
            loss_a_consistency_file.write(str(loss_a_consistency_value_train) + '\n')
            loss_c_file.write(str(loss_c_value_train) + '\n')
            loss_file.write(str(loss_value_train) + '\n')
            seq_file.write(str(batch) + '\n')

        if (batch + 1) % 200 == 0:
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
            checkpoint_path = os.path.join(ckpt_dir, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
            # print("Model saved in file: %s" % filename)

    loss_d_file.close()
    loss_d_consistency_file.close()
    loss_a_file.close()
    loss_a_consistency_file.close()
    loss_c_file.close()
    loss_file.close()
    seq_file.close()


def val():
    print(); print("validating:"); print()

    dimension, angle_set, loss, optimizer, loss_d, loss_c, loss_a, loss_d_consistency, loss_a_consistency = build_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if not ckpt:
        print(); print("checkpoint not found."); print()
    sess.run(tf.global_variables_initializer())
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # result_dir = './result/'
    # if os.path.isdir(result_dir) == True:
        # shutil.rmtree(result_dir)
    # os.mkdir(result_dir)
    # error_dim_file = open('result/error_dim.txt', 'w')
    # error_angle_file = open('result/error_angle.txt', 'w')

    class_accuracy_counter = 0
    mischeck_counter = 0
    dimension_total_error = 0
    angle_total_error = 0
    print("Number of validation data: %d" % data_processing.num_val_data)

    start_time = time.time()

    for i in range(data_processing.num_val_data):
        d_label_ = copy.deepcopy(data_processing.d_epoch_val[i])
        boxsize_ = copy.deepcopy(data_processing.boxsize_epoch_val[i])

        d_value, a_value, loss_d_value = \
            sess.run([dimension, angle_set, loss_d],
                     feed_dict={img: [data_processing.img_epoch_val[i]],
                                boxsize: [boxsize_],
                                d_label: [d_label_],
                                c_label: [data_processing.c_epoch_val[i]],
                                keep_prob: 1.0})

        angle_degree, variance_degree = data_processing.determine_average_degree(a_value, 0)

        dimension_total_error += loss_d_value
        # error_dim_file.write(str(loss_d_value) + '\n')
        angle_error = abs(angle_degree - float(data_processing.a_epoch_val[i][0])*180.0/PI)
        if angle_error > 180:
            angle_error = 360 - angle_error
        angle_total_error += angle_error
        # error_angle_file.write(str(angle_error) + '\n')

    stop_time = time.time()
    # error_dim_file.close()
    # error_angle_file.close()
    print()
    print("dimension average mean squared error: %g" % (dimension_total_error/data_processing.num_val_data))
    print("angle average error: %g" % (angle_total_error/data_processing.num_val_data))
    print("total time cost: %gs" % round(stop_time - start_time, 2))


def write_test_results():
    print(); print("writing test results:"); print()

    dimension, angle_set, loss, optimizer, loss_d, loss_c, loss_a, loss_d_consistency, loss_a_consistency = build_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if not ckpt:
        print(); print("checkpoint not found."); print()
    sess.run(tf.global_variables_initializer())
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    result_dir = './test_results/'
    if os.path.isdir(result_dir) == True:
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for i in range(7518):

        seqname = str(i).zfill(6)
        img_input = scipy.misc.imread(data_processing.img_path + '../../testing/image_2/' + seqname + '.png')
        label_file = data_processing.result_2d_path + seqname + '.txt'
        result_file = result_dir + seqname + '.txt'

        with open(result_file, 'a') as rf:
            with open(label_file, 'r') as lf:
                for ln in lf.readlines():
                    line_data = ln.strip().split(" ")[:]
                    xmin = int(float(line_data[4]))
                    ymin = int(float(line_data[5]))
                    xmax = int(float(line_data[6]))
                    ymax = int(float(line_data[7]))

                    if line_data[0] == 'Pedestrian' and xmin < xmax and ymin < ymax:
                        img_data = copy.deepcopy(img_input[ymin:ymax+1, xmin:xmax+1])
                        img_data = scipy.misc.imresize(img_data, [224, 224])
                        img_data = img_data.astype(np.float32) / 255.0
                        boxsize_data = np.array([xmax-xmin, ymax-ymin]).astype(np.float32)
                        c_data = data_processing.generate_one_hot_list(data_processing.class_dict[line_data[0]], 2)

                        d_value = sess.run([dimension],
                                  feed_dict={img: [img_data],
                                            boxsize: [boxsize_data],
                                            c_label: [c_data],
                                            keep_prob: 1.0})

                        a_value = sess.run([angle_set],
                                  feed_dict={img: [img_data],
                                            boxsize: [boxsize_data],
                                            c_label: [c_data],
                                            d_label: d_value[0],
                                            keep_prob: 1.0})

                        angle_value, variance_degree = data_processing.determine_average_degree(a_value[0], 0)
                        if variance_degree < 100: variance_degree = 100
                        angle_value = angle_value * PI / 180

                        line_data[8] = str(d_value[0][0][0])
                        line_data[9] = str(d_value[0][0][1])
                        line_data[10] = str(d_value[0][0][2])
                        line_data[3] = str(angle_value)
                        line_data[14] = str(angle_value + np.arctan(float(line_data[11]) / float(line_data[13])))
                        result_line = ' '.join(line_data) + '\n'
                        rf.write(result_line)

        print(i)


if __name__ == "__main__":
    if data_processing.toTrain == 1:
        train()
    elif data_processing.toTrain == 0:
        val()
    else:
        write_test_results()

