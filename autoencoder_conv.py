import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import csv
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
test_images_trans = normalizeImage(test_images)
csvFile = open("test_transfor.csv", "w")
writer2 = csv.writer(csvFile)
writer2.writerows(test_images_trans)


learning_rate_start = 0.001
training_epochs = 10001
batch_size = 128
display_step = 1
examples_to_show = 10


inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

### Encoder
conv1 = tf.layers.conv2d(inputs=inputs_, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
# Now 14x14x32
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
# Now 7x7x32
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
# Now 4x4x16
### Decoder
upsample1 = tf.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 7x7x16
conv4 = tf.layers.conv2d(inputs=upsample1, filters=16, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
upsample2 = tf.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 14x14x16
conv5 = tf.layers.conv2d(inputs=upsample2, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
upsample3 = tf.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 28x28x32
conv6 = tf.layers.conv2d(inputs=upsample3, filters=32, kernel_size=(3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
#Now 28x28x1
# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)
y_true = inputs_
# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
# 定义global_step
global_step = tf.Variable(0, trainable=False)
# 通过指数衰减函数来生成学习率
learning_rate = tf.train.exponential_decay(learning_rate_start, global_step, 100, 0.96, staircase=True)

opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

def getBatchdata(data,label,batchsize,ithbatch):
    return data[ithbatch*batchsize: (ithbatch+1)*batchsize], label[ithbatch*batchsize: (ithbatch+1)*batchsize]


train_cost=[]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=config) as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess, './checkpoint_dir/MyModel', global_step=10000)
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
    # total_batch = int(mnist.train.num_examples / batch_size)  # 总批数

    total_batch = int(len(X_train_minmax) / batch_size)  # 总批数
    for epoch in range(training_epochs):
        for i in range(total_batch):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0

            batch_xs, batch_ys = getBatchdata(X_train_minmax,train_labels,batch_size,i)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: batch_xs.reshape((-1, 28, 28, 1)),
                                                             targets_: batch_xs.reshape((-1, 28, 28, 1))})
        if epoch % display_step == 0:
            train_cost.append(batch_cost)
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(batch_cost))
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

    encode_decode = sess.run(
        decoded, feed_dict={inputs_: test_images_trans[0:examples_to_show].reshape((-1, 28, 28, 1))})
    f, a = plt.subplots(2, 10, figsize=(10, 2))

    # encode_decode = sess.run(
    #     y_pred, feed_dict={X: normalizeImage(test_images[0:examples_to_show])})
    # f, a = plt.subplots(2, 10, figsize=(10, 2))

    for i in range(examples_to_show):
        # a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        # a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

        a[0][i].imshow(np.reshape(test_images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    plt.savefig("testing samples.png")
    plt.show()

    plt.show()
    plt.plot(train_cost)
    plt.title('training loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('training loss curve.png')
    plt.show()

    # encoder_result = sess.run(encoded, feed_dict={inputs_: test_images.reshape((-1, 28, 28, 1))})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=test_labels)
    # plt.colorbar()
    # plt.title('test data cluster')
    # plt.savefig('test data cluster.png')
    # plt.show()
