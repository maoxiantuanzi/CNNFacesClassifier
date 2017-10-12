import os
import skimage.data
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

# load data
def load_data(data_dir):
     # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    i=0
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        # print(label_dir)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".jpg")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(i))
        i+=1
    return images, labels

ROOT_PATH = "/home/pengtt/assignment/MSRA-CFW/Sample"
data_dir = os.path.join(ROOT_PATH, "faces_resize")
X, y= load_data(data_dir)
# print(y)
print("load_data done")

# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
X_train,X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size = 0.2,
                                                   random_state = 0)

print("train_test_split done")

n_train = len(X_train)
n_test = len(X_test)
n_classes = len(set(y_train))
print("Unique Labels: {0}\nTotal Images: {1}".format(n_classes, len(X_train)))

# print(X_train.sha)
# X_test2 = np.array([X_test])
# print(X_test2.shape)
# X_train2=np.array([X_train])
# print(X_train2.shape)
# for image in X_test[1:]:
#     image = np.array([image])
#     #print(image)
#     X_test2 = np.append(X_test2,image,0)
# #X_train2 = np.stack((X_train2, x, )),0)
# X_train2=np.array([X_train2])
# print(X_test2.shape)

X_test2 = np.array([X_test])
X_test = X_test2[0]
# print(X_test.shape)
X_train2 = np.array([X_train])
X_train = X_train2[0]
# print(X_train.shape)

X_test2=X_test[:,:,:,np.newaxis]
print(X_test2.shape)
X_train2=X_train[:,:,:,np.newaxis]
print(X_train2.shape)

X_train = X_train2
X_test = X_test2

print('done')

y_train=np.array(y_train)
print(y_train[1000:1500])

# train_mean=np.mean(X_train)
# test_mean=np.mean(X_test)
# print(train_mean)
# print(test_mean)
max_train=X_train.max()
min_train=X_train.min()

max_test=X_test.max()
min_test=X_test.min()

X_train_normalized = (X_train - min_train)/(max_train - min_train)
X_test_normalized = (X_test - min_test)/(max_test - min_test)

X_train2 = (X_train_normalized - 0.5)/0.5
X_test2 = (X_test_normalized - 0.5)/0.5

# print(X_train.max())
# print(X_test.min())

# print(np.var(X_train))
# print(np.var(X_test))

## Normalize the train and test datasets to (-1,1)

# X_train_normalized = (X_train - 0.5)/0.5
# X_test_normalized = (X_test - 0.5)/0.5

print(np.mean(X_train2))
print(np.mean(X_test2))

# X_train, y_train = shuffle(X_train2, y_train)
X_train=X_train2

# CNN
EPOCHS = 60
BATCH_SIZE = 128

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 57x47x1. Output = 53x43x5.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma)) # weight1, 初始化权重, 57-5+1=53
    # W1.shape: [filter_height, filter_width, in_channels, out_channels]
    # 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    #
    # x：为需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
    # 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
    # 注意这是一个4维的Tensor，要求类型为float32和float64其中之一
    #
    # strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
    #
    # padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID') # 实现卷积
    b1 = tf.Variable(tf.zeros(6)) # bias1
    x = tf.nn.bias_add(x, b1) # How does it work? the shape doesn't change
    print("layer 1 shape:",x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 53x43x5. Output = 26x21x5.
    # ksize: The size of the window for each dimension of the input tensor.
    # strides: The stride of the sliding window for each dimension of the input tensor.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 22x17x10.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16))
    x = tf.nn.bias_add(x, b2)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 22x17x10. Output = 11x8x10.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


     # TODO: Layer 2-: Convolutional. Output = 22x17x10.
    W2_ = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean = mu, stddev = sigma))
    x = tf.nn.conv2d(x, W2_, strides=[1, 1, 1, 1], padding='VALID')
    b2_ = tf.Variable(tf.zeros(32))
    x = tf.nn.bias_add(x, b2_)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 22x17x10. Output = 11x8x10.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    # TODO: Flatten. Input = 11x8x10. Output = 880.
    x = flatten(x)

    # TODO: Layer 3: Fully Connected. Input = 880. Output = 120.
    W3 = tf.Variable(tf.truncated_normal(shape=(512, 120), mean = mu, stddev = sigma))
    b3 = tf.Variable(tf.zeros(120))
    x = tf.add(tf.matmul(x, W3), b3)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    W4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    b4 = tf.Variable(tf.zeros(84))
    x = tf.add(tf.matmul(x, W4), b4)

    # TODO: Activation.
    x = tf.nn.relu(x)

    # Dropout
    x = tf.nn.dropout(x, keep_prob)
    print("x shape:",x.get_shape())
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 40.
    W5 = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros(10))
    logits = tf.add(tf.matmul(x, W5), b5)

    return logits

print('done')


tf.reset_default_graph()

x = tf.placeholder(tf.float32, (None, 64, 64, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) # probability to keep units
one_hot_y = tf.one_hot(y, 10)

print('done')


rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate) #adam optimizer
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}) #keep_prob?
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

print('done')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        train_data1, train_label = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        validation_accuracy = evaluate(X_test2, y_test)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print("Model saved")
