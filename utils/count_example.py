#! /bin/env python3
# coding: utf-8

# This script counts instances in a TF Record file

import sys
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

filename = sys.argv[1]

filenames = [filename]
d = tf.data.TFRecordDataset(filenames)
counter = 0
for el in d:
    counter += 1

print(counter)
