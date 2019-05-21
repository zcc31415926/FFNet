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


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PI = 3.141592654

ckpt_dir = './checkpoint/'


# input image: the same size as that in standard VGG16
img = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
# width and length of the object's 2D bounding box
boxsize = tf.placeholder(tf.float32, shape = [None, 2])
# dimension (height, width and length of the object's 3D bounding box) label
d_label = tf.placeholder(tf.float32, shape = [None, 3])
# class label
c_label = tf.placeholder(tf.float32, shape = [None, 2])
# orientation angle label
a_label = tf.placeholder(tf.float32, shape = [None, 1])


def build_model():

    def LeakyReLU(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
    
    # MultiBin angle loss (without classification)
    def determine_angle_loss(angle_prediction, angle_label, index):
        error_total = 0
        angle_tmp = -PI*index/2
        for i in range(data_processing.batch_size):
            sin_prediction = angle_prediction[i][0]
            cos_prediction = angle_prediction[i][1]
            # PI/4: 45degree shift of every bin
            sin_label = tf.sin(angle_label[i][0] + PI/4 + angle_tmp)
            cos_label = tf.cos(angle_label[i][0] + PI/4 + angle_tmp)
            error_total += (1-sin_label*sin_prediction-cos_label*cos_prediction)
        return error_total / data_processing.batch_size

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
        confidence = slim.dropout(confidence, 0.8, scope='dropout_c1')
        confidence = slim.fully_connected(confidence, 2, activation_fn=None, scope='fc_c2')
        loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=c_label, logits=confidence))
        
        confidence_feedback = slim.fully_connected(c_label, 256, activation_fn=None, scope='fc_cfb1')
        # confidence_feedback = slim.fully_connected(confidence, 256, activation_fn=None, scope='fc_cfb1')
        confidence_feedback = LeakyReLU(confidence_feedback, 0.1)
        confidence_feedback = slim.fully_connected(confidence_feedback, 2048, activation_fn=None, scope='fc_cfb2')

        conv_img_concat_d = tf.concat([conv_img, confidence_feedback], 1)

        dimension = slim.fully_connected(conv_img_concat_d, 512, activation_fn=None, scope='fc_d1')
        dimension = LeakyReLU(dimension, 0)
        dimension = slim.dropout(dimension, 0.5, scope='dropout_d1')
        dimension = slim.fully_connected(dimension, 3, activation_fn=None, scope='fc_d2')
        loss_d = tf.losses.mean_squared_error(d_label, dimension)

        dimension_feedback = slim.fully_connected(d_label, 256, activation_fn=None, scope='fc_dfb1')
        # dimension_feedback = slim.fully_connected(dimension, 256, activation_fn=None, scope='fc_dfb1')
        dimension_feedback = LeakyReLU(dimension_feedback, 0)
        dimension_feedback = slim.fully_connected(dimension_feedback, 2048, activation_fn=None, scope='fc_dfb2')

        conv_img_concat_iou = tf.concat([conv_img, dimension_feedback], 1)
        conv_img_concat_a = tf.concat([conv_img_concat_iou, net_boxsize], 1)

        iou = slim.fully_connected(conv_img_concat_iou, 512, activation_fn=None, scope='fc_iou1')
        iou = LeakyReLU(iou, 0)
        iou = slim.dropout(iou, 0.5, scope='dropout_iou1')
        iou = slim.fully_connected(iou, 1, activation_fn=None, scope='fc_iou2')
        min_d = tf.math.minimum(d_label, dimension)
        intersection = tf.reduce_prod(min_d, axis=1)
        box_capacity = tf.reduce_prod(dimension, axis=1)
        box_capacity_label = tf.reduce_prod(d_label, axis=1)
        union = tf.subtract(tf.add(box_capacity, box_capacity_label), intersection)
        iou_label = tf.divide(intersection, union)
        iou_label = tf.reshape(iou_label, shape=[tf.size(iou_label), 1])
        # gradient truncation
        tf.stop_gradient(iou_label)
        loss_iou = tf.losses.mean_squared_error(iou_label, iou)

        angle1 = slim.fully_connected(conv_img_concat_a, 512, activation_fn=None, scope='fc_a11')
        angle1 = LeakyReLU(angle1, 0)
        angle1 = slim.dropout(angle1, 0.5, scope='dropout_a11')
        angle1 = slim.fully_connected(angle1, 2, activation_fn=None, scope='fc_a12')
        angle1 = tf.nn.l2_normalize(angle1, axis=1)
        loss_a1 = determine_angle_loss(angle1, a_label, 0)

        angle2 = slim.fully_connected(conv_img_concat_a, 512, activation_fn=None, scope='fc_a21')
        angle2 = LeakyReLU(angle2, 0)
        angle2 = slim.dropout(angle2, 0.5, scope='dropout_a21')
        angle2 = slim.fully_connected(angle2, 2, activation_fn=None, scope='fc_a22')
        angle2 = tf.nn.l2_normalize(angle2, axis=1)
        loss_a2 = determine_angle_loss(angle2, a_label, 1)

        angle3 = slim.fully_connected(conv_img_concat_a, 512, activation_fn=None, scope='fc_a31')
        angle3 = LeakyReLU(angle3, 0)
        angle3 = slim.dropout(angle3, 0.5, scope='dropout_a31')
        angle3 = slim.fully_connected(angle3, 2, activation_fn=None, scope='fc_a32')
        angle3 = tf.nn.l2_normalize(angle3, axis=1)
        loss_a3 = determine_angle_loss(angle3, a_label, 2)

        angle4 = slim.fully_connected(conv_img_concat_a, 512, activation_fn=None, scope='fc_a41')
        angle4 = LeakyReLU(angle4, 0)
        angle4 = slim.dropout(angle4, 0.5, scope='dropout_a41')
        angle4 = slim.fully_connected(angle4, 2, activation_fn=None, scope='fc_a42')
        angle4 = tf.nn.l2_normalize(angle4, axis=1)
        loss_a4 = determine_angle_loss(angle4, a_label, 3)

        # if number of bins in MultiBin > 4, add more branches here

        angle_set = [angle1, angle2, angle3, angle4]
        loss_a = loss_a1 + loss_a2 + loss_a3 + loss_a4

        total_loss = loss_c + loss_d + loss_a + loss_iou
        optimizer = tf.train.GradientDescentOptimizer(data_processing.learning_rate).minimize(total_loss)

        return dimension, angle_set, confidence, iou, total_loss, optimizer, loss_d, loss_c, loss_a, loss_iou


def train():
    print(); print("training:"); print()
    print("Number of training data: %d" % data_processing.num_train_data)
    print("Number of validation data: %d" % data_processing.num_test_data)

    dimension, angle_set, confidence, iou, total_loss, optimizer, loss_d, loss_c, loss_a, loss_iou = build_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if not ckpt:
        print(); print("checkpoint not found."); print()
    sess.run(tf.global_variables_initializer())
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for epoch in range(data_processing.num_epoch):

        for i in range(int(data_processing.num_train_data/data_processing.batch_size)):
            img_batch, boxsize_batch, d_batch, c_batch, a_batch = data_processing.get_train_data(data_processing.batch_size)

            _, total_loss_output = sess.run([optimizer, total_loss],
                                            feed_dict={img: img_batch,
                                                       boxsize: boxsize_batch,
                                                       d_label: d_batch,
                                                       c_label: c_batch,
                                                       a_label: a_batch})
            
            # prevent gradient explosion
            if math.isnan(total_loss_output):
                sys.exit(0)

            if i % 100 == 0:
                img_test_batch, boxsize_test_batch, d_test_batch, c_test_batch, a_test_batch = \
                    data_processing.extract_random_test_batch(data_processing.batch_size)
                loss_d_value_test, loss_c_value_test, loss_a_value_test, loss_iou_value_test = \
                    sess.run([loss_d, loss_c, loss_a, loss_iou],
                            feed_dict={img: img_test_batch,
                                        boxsize: boxsize_test_batch,
                                        d_label: d_test_batch,
                                        c_label: c_test_batch,
                                        a_label: a_test_batch})
                loss_d_value_train, loss_c_value_train, loss_a_value_train, loss_iou_value_train = \
                    sess.run([loss_d, loss_c, loss_a, loss_iou],
                            feed_dict={img: img_batch,
                                        boxsize: boxsize_batch,
                                        d_label: d_batch,
                                        c_label: c_batch,
                                        a_label: a_batch})

                print("Epoch: %d, Batch: %d:" % (epoch, i))
                print("TEST BATCH: loss_d: %g, loss_c: %g, loss_a: %g, loss_iou: %g" % (loss_d_value_test, loss_c_value_test, loss_a_value_test, loss_iou_value_test))
                print("TRAIN BATCH: loss_d: %g, loss_c: %g, loss_a: %g, loss_iou: %g" % (loss_d_value_train, loss_c_value_train, loss_a_value_train, loss_iou_value_train))

            if i % 100 == 0:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                checkpoint_path = os.path.join(ckpt_dir, "model.ckpt")
                filename = saver.save(sess, checkpoint_path)
                # print("Model saved in file: %s" % filename)


def test():
    print(); print("testing:"); print()
    print("Number of testing data: %d" % data_processing.num_test_data)

    dimension, angle_set, confidence, iou, total_loss, optimizer, loss_d, loss_c, loss_a, loss_iou = build_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if not ckpt:
        print(); print("checkpoint not found."); print()
    sess.run(tf.global_variables_initializer())
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    misclassification_counter = 0
    dimension_total_error = 0
    angle_total_error = 0

    start_time = time.time()
    for i in range(data_processing.num_test_data):

        # orientation determination requires dimension output

        loss_d_value, c_value, d_value, iou_value = \
            sess.run([loss_d, confidence, dimension, iou],
                     feed_dict={img: [data_processing.img_epoch_test[i]],
                                boxsize: [data_processing.boxsize_epoch_test[i]],
                                d_label: [data_processing.d_epoch_test[i]],
                                c_label: [data_processing.c_epoch_test[i]]})
        
        a_value = sess.run([angle_set], 
                           feed_dict={img: [data_processing.img_epoch_test[i]],
                                      boxsize: [data_processing.boxsize_epoch_test[i]],
                                      d_label: d_value[0]})

        dimension_total_error += loss_d_value

        angle_degree, variance_degree = data_processing.determine_average_degree(a_value, 0)
        angle_error = abs(angle_degree - float(data_processing.a_epoch_test[i][0])*180.0/PI)
        if angle_error > 180:
            angle_error = 360 - angle_error
        angle_total_error += angle_error
        
        class_prediction = np.argmax(c_value[0])
        if class_prediction != data_processing.class_dict[data_processing.label_epoch_test[i][0]]:
            misclassification_counter += 1
    
    stop_time = time.time()
    print()
    print("classification accuracy: %g" % (1 - misclassification_counter/data_processing.num_test_data))
    print("dimension average mean squared error: %g" % (dimension_total_error/data_processing.num_test_data))
    print("angle average error: %g" % (angle_total_error/data_processing.num_test_data))
    print("total time cost: %gs" % round(stop_time - start_time, 2))


def write_test_results():
    print(); print("writing test results:"); print()

    label_2D_dir = './label_2d/'
    result_dir = './test_results/'
    # if os.path.isdir(result_dir) == True:
    #     shutil.rmtree(result_dir)
    # os.mkdir(result_dir)

    dimension, angle_set, confidence, iou, total_loss, optimizer, loss_d, loss_c, loss_a, loss_iou = build_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if not ckpt:
        print(); print("checkpoint not found."); print()
    sess.run(tf.global_variables_initializer())
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(7518):

        seqname = str(i).zfill(6)
        img_input = scipy.misc.imread(data_processing.img_path + 'testing/image_2/' + seqname + '.png')
        label_file = label_2D_dir + seqname + '.txt'
        result_file = result_dir + seqname + '.txt'

        with open(result_file, 'a') as rf:
            with open(label_file, 'r') as lf:
                for ln in lf.readlines():
                    line_data = ln.strip().split(" ")[:]
                    xmin = int(float(line_data[4]))
                    ymin = int(float(line_data[5]))
                    xmax = int(float(line_data[6]))
                    ymax = int(float(line_data[7]))

                    if line_data[0] == 'Car' and xmin < xmax and ymin < ymax:
                    # if line_data[0] == 'Pedestrian' and xmin < xmax and ymin < ymax:
                    # if line_data[0] == 'Cyclist' and xmin < xmax and ymin < ymax:
                        img_sample = copy.deepcopy(img_input[ymin:ymax+1, xmin:xmax+1])
                        img_sample = scipy.misc.imresize(img_sample, [224, 224]).astype(np.float32)
                        boxsize_sample = np.array([xmax-xmin, ymax-ymin]).astype(np.float32)
                        class_sample = data_processing.generate_one_hot_list(data_processing.class_dict[line_data[0]], 2)

                        d_value = sess.run([dimension],
                                  feed_dict={img: [img_sample],
                                            boxsize: [boxsize_sample],
                                            c_label: [class_sample]})

                        a_value = sess.run([angle_set],
                                  feed_dict={img: [img_sample],
                                            boxsize: [boxsize_sample],
                                            c_label: [class_sample],
                                            d_label: d_value[0]})
                        
                        angle_degree, variance_degree = data_processing.determine_average_degree(a_value[0], 0)
                        angle_degree = angle_degree * PI / 180

                        line_data[8] = str(d_value[0][0][0])
                        line_data[9] = str(d_value[0][0][1])
                        line_data[10] = str(d_value[0][0][2])
                        line_data[3] = str(angle_degree)
                        line_data[14] = str(angle_degree + np.arctan(float(line_data[11]) / float(line_data[13])))
                        # if there is any confidence modification, change line_data[15] here
                        result_line = ' '.join(line_data) + '\n'
                        rf.write(result_line)

        print('Sample', seqname, 'finished !')


if __name__ == "__main__":
    if data_processing.toTrain == 1:
        train()
    elif data_processing.toTrain == 0:
        test()
    else:
        write_test_results()