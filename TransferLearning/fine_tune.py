#!/usr/bin/env python
# Trains and tests the model and outputs the mean accuracy results to a txt file

import os
import numpy as np
import tensorflow as tf
import math
from datetime import datetime
from oxfordnet import Oxford17Net
from prepare_data import return_data_splits, read_images_from_disk

def train(split_num):
    """
    Function for fine-tuning the Oxford17 dataset on training data.
    
    Attributes:
        split_num: which split to test on, either 1, 2, or 3
    """
        
    # Learning params
    num_epochs = 10
    batch_size = 68

    # Network params
    dropout_rate = 0.5
    num_classes = 17
    # We'll just fine tune the final FC layer, but we could also use 'fc6' and 'fc7' here
    train_layers = ['fc8'] 

    # Path to data splits
    data_path = './data/'
    # Train on a split of the data
    train_paths, train_labels = return_data_splits(data_path, split='trn', split_num=split_num)

    # Get the number of trainingsteps per epoch
    train_batches_per_epoch = np.floor(len(train_paths) / batch_size).astype(np.int16) # 10

    # Convert to Tensorflow data types
    train_image_paths = tf.convert_to_tensor(train_paths, dtype=tf.string)
    train_image_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)

    # create input queues
    train_input_queue = tf.train.slice_input_producer(
                                        [train_image_paths, train_image_labels],
                                        shuffle=True)

    # Load images from disk
    X_train, Y_train = read_images_from_disk(train_input_queue)

    # collect batches of images before processing
    x, y = tf.train.batch(
                                        [X_train, Y_train],
                                        batch_size=batch_size
                                        #,num_threads=1
                                        )

    # Placeholder for dropout probability and learning rate
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # Load the network weights here for use in the model
    net_data = np.load(open("./caffe/oxford102.npy", "rb"), encoding="latin1").item()

    # Initialize model
    model = Oxford17Net(x, num_classes, keep_prob, net_data)

    # Link variable to model output
    score = model.fc8

    # Cross-entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))

    # Find the trainable weights and biases here    
    tvars = []
    for layer in train_layers:
        tvars.append([var for var in tf.trainable_variables() if layer in var.name])
    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=tvars)

    # Accuracy of the model
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Save checkpoints
    checkpoint_path = "./checkpoints/"
    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
    saver = tf.train.Saver()

    # Now train our model
    with tf.Session() as sess:

        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # (optional) learning rate decay
        max_learning_rate = 0.001
        min_learning_rate = 0.0001
        decay_speed = 200.0
        
        # Loop over number of epochs
        for epoch in range(num_epochs):

              print("{} Epoch number: {}".format(datetime.now(), epoch+1))

              step = 1                
                
              while step <= train_batches_per_epoch:
            
                  lr = max_learning_rate
                  # uncomment below if using learning rate decay
                  # lr = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-(epoch*train_batches_per_epoch+step-1)/decay_speed)
                    
                  # Update the weights using backprop
                  sess.run(trainer, feed_dict={keep_prob: dropout_rate, learning_rate: lr})

                  # Check model performance on the batch
                  #l, a = sess.run([loss, accuracy], feed_dict={keep_prob: 1.})
                  l, a = sess.run([loss, accuracy], feed_dict={keep_prob: 1., learning_rate: lr})
                        
                  #print("Loss: %f,     Accuracy: %f" % (l, a))
                  print("Learning Rate: %f,     Loss: %f,     Accuracy: %f" % (lr, l, a))

                  step += 1

        # Save model checkpoint
        checkpoint_name = os.path.join(checkpoint_path, 'model' \
                                       + '_split_' + str(split_num) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)  

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()

def test(split_num, split='tst'):
    """
    Function for testing the fine-tuned Oxford17 dataset on either the validation or test set.
    
    Args:
        split_num: which split to test on, either 1, 2, or 3
        split: which split to test, either 'val' for validation or 'tst' for testing
    Returns:
        Saves a model checkpoint in the /checkpoints directory which can be re-loaded
        to evaluate the performance of the model
    """

    # Network params
    num_classes = 17
    keep_prob = 1.

    # Path to data splits
    data_path = './data/'
    # Use split 1 to test
    paths, labels = return_data_splits(data_path, split=split, split_num=split_num)
    
    # Convert to Tensorflow data types
    image_paths = tf.convert_to_tensor(paths, dtype=tf.string)
    image_labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    batch_size = len(paths)

    # create input queues
    input_queue = tf.train.slice_input_producer(
                                        [image_paths, image_labels],
                                        shuffle=False)

    # Load images from disk
    X, Y = read_images_from_disk(input_queue)

    # collect batches of images before processing
    x, y = tf.train.batch(
                                        [X, Y],
                                        batch_size=batch_size
                                        #,num_threads=1
                                        )

    # Load the network weights here for use in the model
    net_data = np.load(open("./caffe/oxford102.npy", "rb"), encoding="latin1").item()

    # Initialize model
    model = Oxford17Net(x, num_classes, keep_prob, net_data)

    # Link variable to model output
    score = model.fc8

    # Cross-entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))

    # Accuracy of the model
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

    # Now test our model
    with tf.Session() as sess:

        # initialize the variables
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, './checkpoints/model' \
                      + '_split_' + str(split_num) + '.ckpt')

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Get the model accuracy
        a =  sess.run(accuracy)
        print("Accuracy: %f" % a)

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()
        
        # return the model accuracy
        return a

# Write our results to a txt file
f= open("results.txt","a+")

# Store accuracies from each split of the data
accuracy = []

# Loop through the three splits
for split in xrange(3):
    
    f.write("-"*7)
    f.write("\nSplit " + str(split+1) + "\n")
    f.write("-"*7)
    
    # Reset the Tensorflow graph between training and testing
    tf.reset_default_graph()
    
    # Train on a split
    train(split+1)
    
    # Reset the Tensorflow graph between training and testing
    tf.reset_default_graph()
    
    # Append the accuracy on this split so that we can compute mean accuracy
    a = test(split+1)
    f.write("\nTest Accuracy: %s" % (a) + '\n\n\n')
    accuracy.append(a)

# Overall mean accuracy
f.write("Mean Accuracy: %s" % str(np.mean(accuracy)))
f.close()