# -*- coding: utf-8 -*-

import os
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim import nets
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  ###
import cv2
import pdb
import heapq

n_train = 1000000
n_test = 10000

topk = 3

####  Parameter Settings

rows, cols, channels = 224, 224, 3
training_epochs = 200
num_classes = 2019
learn_rate = 1e-3
weight_decay = 0.0005
batch_size = 32

save_model_path = './model/kaggle_model.ckpt'
init_resnet_model_file = './resnet50_tf/resnet_v1_50.ckpt'
train_log_dir = './train_log/'
test_log_dir = './test_log/'


train_data_path = r'./traintfrecord/'
train_data_files = os.listdir(train_data_path)
for i in range(len(train_data_files)):
    train_data_files[i] = os.path.join(train_data_path,train_data_files[i])
print("train_data_files :",train_data_files)

test_data_path = r'./testtfrecord/'
test_data_files = os.listdir(test_data_path)
for i in range(len(test_data_files)):
    test_data_files[i] = os.path.join(test_data_path,test_data_files[i])
print("test_data_files :",test_data_files)

def accuracy(labels,logits):
    acc = tf.nn.in_top_k(logits[0],labels[0],0)
    acc = tf.cast(acc, tf.float32)
    return tf.reduce_mean(acc)

def read_tfrecords(data_files):
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    
    
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                           'img_width': tf.FixedLenFeature([], tf.int64),
                                           'img_height': tf.FixedLenFeature([], tf.int64),
                                       })  #取出包含image和label的feature对象

    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    height = tf.cast(features['img_height'],tf.int32)
    width = tf.cast(features['img_width'],tf.int32)
    label = tf.cast(features['label'], tf.int32)
    channel = 3
    image = tf.reshape(image, [224, 224, 3])
    image = tf.image.per_image_standardization(image)
    return image,height,width,label

def get_batch(batchsize,data_files):

    image, height, width, label = read_tfrecords(data_files)
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 5 * batchsize
    # print("batchsize :",batchsize)
    img_batch, label_batch = tf.train.shuffle_batch([image,label], batch_size=batchsize, capacity=capacity, num_threads=2, min_after_dequeue=min_after_dequeue)

    label_batch = tf.reshape(label_batch,[batch_size,])
    return img_batch, label_batch


def main():

    # train_image_batch, train_label_batch = get_batch(batchsize=batch_size, data_files = train_data_files)
    train_image_batch, train_label_batch = get_batch(batchsize=batch_size, data_files = test_data_files)
    test_image_batch , test_label_batch = get_batch(batchsize=batch_size, data_files = test_data_files)

    inputs = tf.placeholder(tf.float32, shape=[batch_size, rows, cols, channels], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[batch_size,], name='labels')
    is_training = tf.placeholder(tf.bool, name='is_train')
    graph = tf.get_default_graph()

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(inputs, num_classes=None, global_pool=True, is_training=is_training)

    variables_to_restore = slim.get_variables_to_restore(exclude=["logits", "predictions", "SpatialSqueeze"])
    global_steps = tf.Variable(0, trainable=False)
    net = slim.flatten(net)
    pred = slim.fully_connected(net, num_classes, scope='fc')
    pred_after = tf.nn.softmax(pred, name='softmax')

    # pred = tf.nn.softmax(logits, name='softmax')
    classes = tf.argmax(pred_after, axis=1, name='prediction')
    # correct = tf.equal(tf.cast(classes, dtype=tf.int32), tf.argmax(labels, axis=1))  # 返回一个数组 表示统计预测正确或者错误
    # accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 求准确率
    # pdb.set_trace()
    correct = tf.equal(tf.cast(classes, dtype=tf.int64), tf.argmax(labels, axis=0))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 求准确率

    # pdb.set_trace()
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=pred))
    # l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    # total_loss = loss + l2_loss * weight_decay
    total_loss = loss
    # learning_rate = tf.train.exponential_decay(learning_rate=learn_rate, global_step=global_steps, decay_steps=10000, decay_rate=0.5, staircase=False)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    learning_rate = tf.train.exponential_decay(learn_rate, global_steps, 5000, 0.9)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # learning_rate = tf.train.exponential_decay(learn_rate, global_steps, 5000, 0.9)
    # learning_rate = tf.convert_to_tensor(learn_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate)


    train_step = optimizer.minimize(total_loss, global_step=global_steps)

    saver_restore = tf.train.Saver(var_list=variables_to_restore)
    saver = tf.train.Saver(tf.global_variables())

    # init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True)

    train_batch = int(np.ceil(n_train / batch_size))
    test_batch = int(np.ceil(n_test / batch_size))

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # 检查最近的检查点文件
        # ckpt = tf.train.latest_checkpoint(save_model_path)
        # print("ckpt :",ckpt)

        ckpt = tf.train.get_checkpoint_state('./model/')
        print(ckpt)

        if ckpt and ckpt.model_checkpoint_path:
            print("ckpt :",ckpt)
            print("ckpt.model_checkpoint_path :",ckpt.model_checkpoint_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
            start_epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("Successfully load model from save path: %s and epoch: %s"% (ckpt.model_checkpoint_path, start_epoch))
        else :
            # print("Training from scratch")
            saver_restore.restore(sess, init_resnet_model_file)
            print('从官方模型加载训练！')


        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # if ckpt != None:
        #     saver.restore(sess, ckpt)
        #     print('从上次训练保存后的模型继续训练！')
        # else:
        #     saver_restore.restore(sess, init_resnet_model_file)
        #     print('从官方模型加载训练！')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_dir)

        # 保存模型参数，用来进行可视化
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learn rate', learning_rate)
        merged_summary = tf.summary.merge_all()
        print('开始训练！')

        for epoch in range(1):
            train_cost = 0.0
            train_acc = 0.0
            test_cost = 0.0
            test_acc = 0.0
            
            totalnum = 0
            cnt = 0
            for i in range(test_batch):
                # print("i :",i)
                reusltlist = []
                train_images, train_labels = sess.run([train_image_batch, train_label_batch])
                train_dict = {inputs: train_images, labels: train_labels, is_training: True}
                classes_val,pred_after_val = sess.run([classes,pred_after], feed_dict=train_dict)

                for index in range(batch_size):
                	result =  heapq.nlargest(topk,range(len(pred_after_val[index])),pred_after_val[index].take)
                	reusltlist.append(result)

                # print("reusltlist :",reusltlist)
                # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                # print("train_labels :",train_labels)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                for index in range(batch_size):
                	totalnum = totalnum + 1
                	print("totalnum :",totalnum)
                	print(index," ",train_labels[index]," ",reusltlist[index])
                	if train_labels[index] in reusltlist[index]:
                		# print("true :",train_labels[index])
                		cnt = cnt + 1

        acc = cnt*1.0/totalnum*1.0
        print("acc :",acc)




                # print("classes_val :",classes_val)
                # print("pred_after_val :",pred_after_val)
                # print("train_labels :",train_labels)
                # result_list = np.zeros((batch_size,3))
                # print(result_list[0])


        print("训练完成")

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()















