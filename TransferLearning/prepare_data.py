import tensorflow as tf
from scipy.io import loadmat

def return_data_splits(filepath, split, split_num):
    """Returns the trn, val, or tst split of the data from datasplits.mat
    Args:
      filepath (str): where the data is located
      split (str): which split to use, either 'trn', 'val', or 'tst'
      split_num (int): which split number to use, either 1, 2, or 3
    Returns:
      paths (str): the list of filenames of the images in the data split
      labels (int): the list of int labels corresponding to the images in paths
    """
    data_splits = loadmat(filepath + 'datasplits.mat')
    
    # Resized images are located in the 'resized' directory
    prepend_path = filepath + 'resized/image_'
    
    # Pull out the data from the .mat structure
    paths = [prepend_path + "%04d" % (data) + ".jpg" for data in data_splits[split+str(split_num)][0]]
    labels = [(data-1)/80 for data in data_splits[split+str(split_num)][0]]
    
    return paths, labels

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    IMAGE_HEIGHT = 227
    IMAGE_WIDTH = 227
    NUM_CHANNELS = 3
    NUM_CLASSES = 17
    
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    # Do some additional pre-processing here
    example = tf.to_float(example) # convert from uint8 to float32
    example = example[..., ::-1] # switch from RGB to BGR
    example -= tf.constant([103.939, 116.779, 123.68], dtype=tf.float32) # subtract mean Imagenet pixel values    
    example.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    
    label = tf.one_hot(label, NUM_CLASSES) # one hot encoding of labels
    
    return example, label