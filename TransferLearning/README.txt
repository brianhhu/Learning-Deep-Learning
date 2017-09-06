Caffe-Tensorflow Fine Tuning on Oxford17 Flower Dataset
--------


This directory contains all the files need to transform a Caffe model originally
trained on the Oxford102 flower dataset into a Tensorflow model and fine-tune it
on the smaller Oxford17 flower dataset.


Dependencies
------------
    - Python 2.7 (untested on Python 3+)
    - numpy (version 1.12.1)
    - scipy (version 0.19.1)
    - PIL (version 4.2.1)
    - tensorflow (version 1.1.0)


Download the data
-----------------
To download the data and Caffe models, we'll use the following command:


$ python bootstrap.py


The data will be placed in the /data/jpg directory. This file was largely modified from
the following link corresponding to the Oxford102 Caffe model:

https://github.com/jimgoo/caffe-oxford102/blob/master/bootstrap.py


Pre-process the images
----------------------
We batch process the images by resizing and cropping them to the correct size (227x227)
before inputting them into the network. This will produce a new set of images, which we
place in a separate /data/resized directory. To do this, run the following:


$ python batch_process.py ./data/jpg/*.jpg


This was a helpful link for showing how to use the Image package from PIL to do this:

http://www.coderholic.com/batch-image-processing-with-python/


Update the Caffe model files
----------------------------
Before converting our Caffe model to Tensorflow, we'll first have to update the Caffe files (both the
.prototxt and the .caffemodel files). We'll use the upgrade_net_proto_txt and upgrade_net_proto_binary
functions that ship with Caffe (this assumes you have Caffe installed). Use the following commands:


$ (CAFFE_ROOT)/build/tools/upgrade_net_proto_text ./caffe/deploy.prototxt ./caffe/deploy_new.prototxt
$ (CAFFE_ROOT)/build/tools/upgrade_net_proto_binary ./caffe/oxford102.caffemodel ./caffe/oxford102_new.caffemodel


Caffe to Tensorflow Model Conversion
------------------------------------
Then we'll port our Caffe model to tensorflow using the open-source caffe-tensorflow tool
(https://github.com/ethereon/caffe-tensorflow). Run the following command:


$ (CAFFE_TENSORFLOW_ROOT)/convert.py  --caffemodel ./data/oxford102_new.caffemodel \
                                      --data-output-path ./data/oxford102.npy \
                                      ./data/deploy_new.prototxt


We'll only be using the weights (oxford102.npy) from this conversion, as I couldn't get the oxford102.py network
to work with my version of Tensorflow (there was an issue with the tf.split command). Fortunately for us, there
is a readily available Tensorflow port of the AlexNet architecture that seems to work well. I used parts of this
script to build an AlexNet model class (which can be found in oxfordnet.py):

http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet_forward_newtf.py


Fine-Tune the Oxford102 model on the Oxford17 flower dataset
------------------------------------------------------------
To fine-tune the model, simply run the following command:


$ python fine_tune.py


This will fine-tune only the last layer (fc8) of the Oxford102 model. We first train a model, save the model
as a checkpoint (in the checkpoints directory), and then load the model back and evaluate the model on the
test dataset. We also make use of Tensorflow Queues in order to batch load in data for training and testing.
The model achieves a mean accuracy of 91.7% on the test dataset averaged across all three splits of the data.
The final results of our model are found in results.txt.


Future Work
-----------
I didn't do an extensive hyperparameter search of the other parameters of the model (e.g. learning rate, 
batch size, etc.), which could potentially improve performance of the model. Other layers in the model
could also be fine tuned (e.g. fc6 and fc7). Finally, it is also possible to use other techniques such
as model ensembles to improve performance of the model.


Please address any questions to: bhu6@jhmi.edu