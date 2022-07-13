import tensorflow as tf
import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
import numpy as np

import os
import time
import datetime
import sys
import json
import argparse
import pprint as pp
# import yaml

from utils.utils import create_dir, generate_images, CycleGAN, preprocess_image_train, preprocess_image_test
sys.path.insert(1, 'C:\\Users\\kkosara\\Downloads\\')
import Soiled

# A: Unsoiled
# B: Soiled


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    parser = argparse.ArgumentParser(description='provide arguments for cycle GAN implemetation')
    
    # general parameters
    parser.add_argument('--summary_dir', help='directory for saving and loading model and other data', default='./exp')
    parser.add_argument('--seed', help='seed for random number generator', type=int, default=2222)
    parser.add_argument('--data_dir', help='path to the TFrecords directory', default=None)


    # model hyper parameters
    parser.add_argument('--OUTPUT_CHANNELS', help='OUTPUT_CHANNELS for generator', type=int, default=3)
    parser.add_argument('--crop_size', help='crop size for preprocessing the train data', type = json.loads, default=[256, 256, 3])
    parser.add_argument('--LAMBDA', help='lambda loss factor', type=int, default=10)
    parser.add_argument('--chkpoint_step', help='after every chkpoint_step epochs, the model gets saved', type=int, default=10)

    # learning hyper parameters
    parser.add_argument('--buffer_size', help='Buffer size', type=int, default=1000)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)
    parser.add_argument('--epochs', help='Total number of epochs', type=int, default=20)
    parser.add_argument('--AUTOTUNE', help='by default it uses - tf.data.experimental.AUTOTUNE', type=int, default=None)







    args = vars(parser.parse_args())
    if args['AUTOTUNE'] == None:
        args['AUTOTUNE'] =  tf.data.experimental.AUTOTUNE
    np.random.seed(args['seed'])
    args['summary_dir'] = args['summary_dir'] + '/'
    create_dir(args['summary_dir'])
    start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    results_dir = args['summary_dir'] + start_time + '/'
    create_dir(results_dir)

    pp.pprint(args)


    dataset, metadata = tfds.load('Soiled',with_info=True, as_supervised=True)
    
    train_unsoiled, train_soiled = dataset['trainA'], dataset['trainB']
    test_unsoiled, test_soiled = dataset['testA'], dataset['testB']


    train_unsoiled = train_unsoiled.cache().map(preprocess_image_train, num_parallel_calls=args['AUTOTUNE']).shuffle(args['buffer_size']).batch(args['batch_size'])
    train_soiled = train_soiled.cache().map(preprocess_image_train, num_parallel_calls=args['AUTOTUNE']).shuffle(args['buffer_size']).batch(args['batch_size'])
    test_unsoiled = test_unsoiled.map(preprocess_image_test, num_parallel_calls=args['AUTOTUNE']).cache().shuffle(args['buffer_size']).batch(args['batch_size'])
    test_soiled = test_soiled.map(preprocess_image_test, num_parallel_calls=args['AUTOTUNE']).cache().shuffle(args['buffer_size']).batch(args['batch_size'])


    sample_unsoiled_image = next(iter(train_unsoiled))
    sample_soiled_image = next(iter(train_soiled))


    train_dataset = tf.data.Dataset.zip((train_unsoiled, train_soiled))
    cyclegan = CycleGAN(epochs = 10, enable_function=True, path = results_dir)
    cyclegan.train(train_dataset, sample_soiled_image, sample_unsoiled_image, chkpoint_step = args['chkpoint_step'])

