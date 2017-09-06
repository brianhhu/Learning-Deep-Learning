import tensorflow as tf
import numpy as np

class Oxford17Net(object):
    """
    Class file for the Oxford17Net. It creates the Tensorflow structure for the model
    and populates the weights and biases from the Caffe model (caffe-tensorflow).
    
    This code was adapted from the script found here:
    
    http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet_forward_newtf.py
    """
  
    def __init__(self, x, num_classes, keep_prob, net_data):
        """
        Attributes:
            NUM_CLASSES (int): Number of classes to output final classification on
            KEEP_PROB (float): Dropout probability in the fully-connected layers
            WEIGHTS (dict of numpy arrays): Model weights and biases from the caffe-tensorflow conversion
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.WEIGHTS = net_data
    
        # Call the create function to build the computational graph of AlexNet
        self.create()
    
    def create(self):
        """Creates the model from the caffe-tensorflow conversion of AlexNet"""
        # (self.feed('data')
        #         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        #         .lrn(2, 2e-05, 0.75, name='norm1')
        #         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        #         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
        #         .lrn(2, 2e-05, 0.75, name='norm2')
        #         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        #         .conv(3, 3, 384, 1, 1, name='conv3')
        #         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
        #         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
        #         .fc(4096, name='fc6')
        #         .fc(4096, name='fc7')
        #         .fc(1000, relu=False, name='fc8')
        #         .softmax(name='prob'))

        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1W = tf.Variable(self.WEIGHTS["conv1"]["weights"], name='conv1/weights')
        conv1b = tf.Variable(self.WEIGHTS["conv1"]["biases"], name='conv1/biases')
        conv1_in = conv(self.X, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.Variable(self.WEIGHTS["conv2"]["weights"], name='conv2/weights')
        conv2b = tf.Variable(self.WEIGHTS["conv2"]["biases"], name='conv2/biases')
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)

        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.Variable(self.WEIGHTS["conv3"]["weights"], name='conv3/weights')
        conv3b = tf.Variable(self.WEIGHTS["conv3"]["biases"], name='conv3/biases')
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)

        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.Variable(self.WEIGHTS["conv4"]["weights"], name='conv4/weights')
        conv4b = tf.Variable(self.WEIGHTS["conv4"]["biases"], name='conv4/biases')
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)

        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.Variable(self.WEIGHTS["conv5"]["weights"], name='conv5/weights')
        conv5b = tf.Variable(self.WEIGHTS["conv5"]["biases"], name='conv5/biases')
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)

        #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        #fc6
        #fc(4096, name='fc6')
        fc6W = tf.Variable(self.WEIGHTS["fc6"]["weights"], name='fc6/weights')
        fc6b = tf.Variable(self.WEIGHTS["fc6"]["biases"], name='fc6/biases')
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

        #Add a dropout layer here
        dropout6 = tf.nn.dropout(fc6, self.KEEP_PROB)
        
        #fc7
        #fc(4096, name='fc7')
        fc7W = tf.Variable(self.WEIGHTS["fc7"]["weights"], name='fc7/weights')
        fc7b = tf.Variable(self.WEIGHTS["fc7"]["biases"], name='fc7/biases')
        fc7 = tf.nn.relu_layer(dropout6, fc7W, fc7b)

        #Add a dropout layer here
        dropout7 = tf.nn.dropout(fc7, self.KEEP_PROB)
        
        #fc8
        #fc(1000, relu=False, name='fc8')
        fc8W = tf.get_variable("fc8/weights", [4096, self.NUM_CLASSES])
        fc8b = tf.get_variable("fc8/biases", [self.NUM_CLASSES])
        self.fc8 = tf.nn.xw_plus_b(dropout7, fc8W, fc8b)       
  
"""
Predefine convolution operation for the AlexNet
""" 
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    
    Convolution operator using the special tf.split function
    
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])