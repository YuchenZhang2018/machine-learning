import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import csv
from sklearn.preprocessing import LabelBinarizer
# 导入MNIST数据
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
train = pd.read_csv('sign-language-mnist/sign_mnist_train.csv')
test = pd.read_csv('sign-language-mnist/sign_mnist_test.csv')
train_labels = train['label'].values

unique_val = np.array(train_labels)
np.unique(unique_val)

plt.figure(figsize = (18,8))
sns.countplot(x =train_labels)

train.drop('label', axis = 1, inplace = True)

train_images = train.values


def normalizeImage(images):
    min_max_scaler = preprocessing.MinMaxScaler()

    images_trans = np.zeros([len(images), 784], dtype=float)
    for i in range(len(images)):
        images_trans[i] = min_max_scaler.fit_transform(np.reshape(images[i], (28, 28))).flatten()
    return images_trans

print("preprocessign train data")
X_train_minmax = normalizeImage(train_images)
csvFile = open("train_transfor.csv", "w")
writer1= csv.writer(csvFile)
writer1.writerows(X_train_minmax)

test_labels = test['label'].values
test.drop('label',axis=1,inplace = True)
test_images = test.values

print("preprocessign test data")
X_test_minmax = normalizeImage(test_images)
csvFile = open("test_transfor.csv", "w")
writer2 = csv.writer(csvFile)
writer2.writerows(X_test_minmax)


label_binrizer = LabelBinarizer()
train_labels = label_binrizer.fit_transform(train_labels)
test_labels = label_binrizer.fit_transform(test_labels)

learning_rate_start = 0.001
training_epochs = 501
batch_size = 128
display_step = 1
examples_to_show = 10
dropout=0.2
classnum=label_binrizer.classes_.size


inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, classnum), name='targets')

weights = {
    'wd1': tf.get_variable('W3', shape=(4*4*64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128,classnum), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(classnum), initializer=tf.contrib.layers.xavier_initializer()),
}

conv1 = tf.layers.conv2d(inputs=inputs_, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
#now  28*28*64
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now 14x14x64
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x64
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# Now 7x7x64
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=64, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x64
encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
# Now 4x4x64
# Fully connected layer
# Reshape conv2 output to fit fully connected layer input
fc1 = tf.reshape(encoded, [-1, weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, dropout)
# Output, class prediction
# finally we multiply the fully connected layer with the weights and add a bias term.
logits = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

prediction = tf.nn.softmax(logits)

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
# 定义global_step
global_step = tf.Variable(0, trainable=False)
# 通过指数衰减函数来生成学习率
learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, 100, 0.96, staircase=True)

opt = tf.train.AdamOptimizer(0.001).minimize(cost)

y_true = tf.placeholder(tf.float32, shape=[None, classnum], name='y_true')
acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(y_true, axis=1), predictions=tf.argmax(prediction, 1))

def getBatchdata(data,label,batchsize,ithbatch):
    return data[ithbatch*batchsize: (ithbatch+1)*batchsize], label[ithbatch*batchsize: (ithbatch+1)*batchsize]


train_cost=[]
test_acc=[]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=config) as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #     init = tf.initialize_all_variables()
    # else:
    init = tf.global_variables_initializer()
    sess.run(tf.local_variables_initializer())
    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess, './checkpoint_dir/NN', global_step=500)
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
    # total_batch = int(mnist.train.num_examples / batch_size)  # 总批数

    total_batch = int(len(X_train_minmax) / batch_size)  # 总批数
    for epoch in range(training_epochs):
        for i in range(total_batch):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0

            batch_xs, batch_ys = getBatchdata(X_train_minmax,train_labels,batch_size,i)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: batch_xs.reshape((-1, 28, 28, 1)),
                                                             targets_: batch_ys})
        if epoch % display_step == 0:
            train_cost.append(batch_cost)
            test_batch_xs, test_batch_ys = getBatchdata(X_train_minmax,test_labels,71,epoch%100)
            predict = sess.run(
                prediction, feed_dict={inputs_:test_batch_xs.reshape((-1, 28, 28, 1))})
            # pred_rst = tf.equal(tf.argmax(predict, 1), tf.argmax(test_batch_ys, 1))
            accuracy = sess.run(acc_op, feed_dict={y_true:test_batch_ys,inputs_:test_batch_xs.reshape((-1, 28, 28, 1))})
            # acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(predict, 1),
            #                                   predictions=tf.argmax(test_batch_ys, 1))
            # acc = sess.run(tf.reduce_mean(tf.cast(pred_rst, tf.float32)))
            test_acc.append(accuracy)
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(batch_cost),"test acc=","{:.4f}".format(accuracy))
    print("Optimization Finished!")

    # total_batch = int(len(train_images) / batch_size)  # 总批数
    # for epoch in range(training_epochs):
    #     for i in range(total_batch):
    #         # batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
    #
    #         batch_xs, batch_ys = getBatchdata(train_images, train_labels, batch_size, i)
    #         batch_xs_trans = normalizeImage(batch_xs)# max(x) = 1, min(x) = 0
    #         # Run optimization op (backprop) and cost op (to get loss value)
    #         _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs_trans})
    #     if epoch % display_step == 0:
    #         print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    # print("Optimization Finished!")


    # encode_decode = sess.run(
    #     y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})

    predict = sess.run(
        prediction, feed_dict={inputs_: X_test_minmax[0:10].reshape((-1, 28, 28, 1))})
    pred_rst = tf.equal(tf.argmax(predict, 1), tf.argmax(test_labels[0:10], 1))

    acc = sess.run(tf.reduce_mean(tf.cast(pred_rst,tf.float32)))
    print(acc)
    plt.show()
    plt.plot(train_cost,label='trainloss')
    plt.plot(test_acc,label='test accuracy')
    plt.title('training loss and test accuracy curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('training loss and acc curve.png')
    plt.show()
