#!/usr/bin/env python
# Batch image resize script using PIL

"""
This script takes the original size images from the Oxford17 flower dataset
and first rescales them to 256x256 pixels, then performs a center crop to
achieve a final image size of 227x227 pixels. The resized images are placed
in the /data/resized directory

This was a helpful link for doing batch processing of images with PIL:

http://www.coderholic.com/batch-image-processing-with-python/
"""

import sys
import os.path
from PIL import Image

# Resize and crop parameters
resize_size = (256, 256)
crop_size = (227, 227)

# Create new directory for storing resized images
if not os.path.exists('data/resized'):
    os.mkdir('data/resized')

# Loop through all provided arguments
for i in range(1, len(sys.argv)):
    try:
        # Attempt to open an image file
        filepath = sys.argv[i]
        image = Image.open(filepath)
    except IOError, e:
        # Report error, and then skip to the next argument
        print "Problem opening", filepath, ":", e
        continue

    # Resize the image
    image = image.resize(resize_size, Image.ANTIALIAS)
    # left, top, right, bottom
    image = image.crop(((resize_size[1]-crop_size[1])/2, (resize_size[0]-crop_size[0])/2, \
                       (resize_size[1]+crop_size[1])/2, (resize_size[0]+crop_size[0])/2))
    
    # Split our original filename into name and extension
    name = os.path.basename(os.path.normpath(filepath))
    
    # Save resized images in the /data/resized directory
    image.save('./data/resized/' + name)