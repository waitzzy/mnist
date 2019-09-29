import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
examples_to_show = 10
mnist = input_data.read_data_sets('MNIST-data',one_hot=False)
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels


def encode_ont_hot(labels):
    index = (labels == 9)
    labels[index] = 4
    labels_num = labels.shape[0]
    index_offset = np.arange(labels_num) * 5
    labels_one_hot = np.zeros((labels_num, 5))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

train_filter = np.where((Y_train == 0) | (Y_train == 1)|(Y_train == 2) |(Y_train == 3)|(Y_train == 9))
test_filter = np.where((Y_test == 0) | (Y_test == 1)|(Y_test == 2)|(Y_test == 3)|(Y_test == 9))
X_train, Y_train = X_train[train_filter], Y_train[train_filter]
X_test, Y_test = X_test[test_filter], Y_test[test_filter]
Y_train = encode_ont_hot(Y_train)
Y_test = encode_ont_hot(Y_test)
batch = 50
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,5])
target = tf.placeholder(tf.float32, [None,784])
keep_prob = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32)



def accuracy(test_x,test_y,sess,output):
    res_y = sess.run(output,feed_dict={x:test_x,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(res_y,1),tf.argmax(test_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={x:test_x,y:test_y,keep_prob:1})
    return result

def weight(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def MaxPool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


x_img = tf.reshape(x, [-1, 28, 28, 1])
target_img = tf.reshape(x,[-1, 28, 28, 1])
# L1
w1 = weight([5,5,1,64])
b1 = bias([64])
temp = conv2d(x_img, w1)
activation = tf.nn.relu(temp + b1)
# pool1
pool1 = MaxPool(activation)
# L2
w2=weight([14*14*64,1000])
b2=bias([1000])
tempx = tf.reshape(pool1, [-1, 14*14*64])
activation=tf.nn.relu(tf.matmul(tempx,w2)+b2)
drop = tf.nn.dropout(activation, keep_prob)
# L3(1)
w3 = weight([1000,5])
b3 = bias([5])
output1 = tf.nn.softmax(tf.matmul(drop, w3) + b3)
loss1 = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output1) ,reduction_indices=[1]))
train1 = tf.train.AdamOptimizer(1e-4).minimize(loss1)

# L3(2)

decov_w3 = weight([1000,14*14*64])
deconv_b3 = bias([14*14*64])
activation_3=tf.nn.relu(tf.matmul(drop,decov_w3)+deconv_b3)
drop_3 = tf.nn.dropout(activation_3, keep_prob)
fold = tf.reshape(drop_3,[-1,14,14,64])
#w4 = weight([5,5,1,64])
deconv = tf.layers.conv2d_transpose(fold, 64, [5, 5], strides=2, padding='SAME')
#deconv = tf.nn.conv2d_transpose(fold,w4,output_shape=[batch_size,28,28,64],strides=[1,2,2,1],padding="SAME")
w4 = weight([5,5,64,1])
b4 = bias([1])
reimg = conv2d(deconv,w4)
reimg = tf.nn.bias_add(reimg, b4)
decoded = tf.sigmoid(reimg)
cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=reimg, labels=target_img)
#loss2 = tf.nn.l2_loss(target_img - reimg)
loss2 = tf.reduce_mean(cost)
train2 = tf.train.AdamOptimizer(0.001).minimize(loss2)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    test_filter = np.where((Y_test == 0) | (Y_test == 1) | (Y_test == 2) | (Y_test == 3) | (Y_test == 9))
    X_test, Y_test = X_test[test_filter], Y_test[test_filter]
    Y_test = encode_ont_hot(Y_test)
    yyy = np.zeros(200)
    for i in range(500):
        X_batch,Y_batch = mnist.train.next_batch(50)
        X_train = X_batch
        Y_train = Y_batch
        train_filter = np.where((Y_train == 0) | (Y_train == 1) | (Y_train == 2) | (Y_train == 3) | (Y_train == 9))
        X_train, Y_train = X_train[train_filter], Y_train[train_filter]
        Y_train = encode_ont_hot(Y_train)
        batch_cost, _= sess.run([loss2, train2],feed_dict={x:X_train,target:X_train,keep_prob:0.6,batch_size:X_train.shape[0]})
        #batch_cost2, _= sess.run([loss1, train1],feed_dict={x:X_train,y:Y_train,keep_prob:0.6,batch_size:X_train.shape[0]})
        #print("Epoch: {}/{}...".format(i, 200),
        #      "Training loss: {:.4f}".format(batch_cost))
    encode_decode = sess.run(
    decoded, feed_dict={x: X_test[:50],keep_prob:0.6})
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(X_test[i+20], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i+20], (28, 28)))
    plt.show()
    '''
            yyy[i] = batch_cost1+batch_cost2
    xxx = np.arange(0, 200, 1)
    plt.plot(xxx, yyy, "b-", label="loss")
    plt.title("total loss")
    plt.xlabel("curves", fontsize=15)
    plt.ylabel("$loss$", fontsize=15)
    plt.show()
            if(i%5==0):
            j=i/50
            j = int(j)
            result = accuracy(X_test,Y_test,sess,output1)
            print(result)
            yyy[j] = result

    '''


