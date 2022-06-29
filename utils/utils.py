import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

import os
import time
import sys
import json
import argparse
import pprint as pp
# import yaml
import datetime


import os
import time
from absl import app

import tensorflow as tf

def create_dir(dir_name:str):
    try:
        os.mkdir(dir_name)
    except:
        print("Sub directory {} is already available".format(dir_name))
        pass

def generate_images(model, test_input):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


@tf.function
def preprocess_image_train(image, label, crop_size = [256, 256, 3]):
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, size=crop_size)
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

@tf.function
def preprocess_image_test(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
  """Downsamples an input.
  Conv2D => Batchnorm => LeakyRelu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer
  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def unet_generator(output_channels, norm_type='batchnorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
  Returns:
    Generator model
  """

  down_stack = [
      downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
      downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
      downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
      downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
      downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
      downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
      downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
      downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
  ]

  up_stack = [
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
      upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
      upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
      upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
      upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 4, strides=2,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, 3])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(norm_type='batchnorm', target=True):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.
  Returns:
    Discriminator model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
  x = inp

  if target:
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
  down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(
      1, 4, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  if target:
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
  else:
    return tf.keras.Model(inputs=inp, outputs=last)


def get_checkpoint_prefix():
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

  return checkpoint_prefix


class CycleGAN(object):
    """CycleGAN class.
    Args:
        epochs: Number of epochs.
        enable_function: If true, train step is decorated with tf.function.
        buffer_size: Shuffle buffer size..
        batch_size: Batch size.
        lambda: lambda loss factor
    """
    def __init__(self, epochs, enable_function, path, lambda_value = 10, OUTPUT_CHANNELS = 3):
        self.epochs = epochs
        self.enable_function = enable_function
        self.lambda_value = lambda_value
        self.OUTPUT_CHANNELS = OUTPUT_CHANNELS
        self.path = path

        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        self.generator_g = unet_generator(self.OUTPUT_CHANNELS, norm_type='instancenorm')
        self.generator_f = unet_generator(self.OUTPUT_CHANNELS, norm_type='instancenorm')

        self.discriminator_x = discriminator(norm_type='instancenorm', target=False)
        self.discriminator_y = discriminator(norm_type='instancenorm', target=False)

        self.checkpoint = tf.train.Checkpoint(
            generator_g=self.generator_g,
            generator_f=self.generator_f,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y,
            generator_g_optimizer=self.generator_g_optimizer,
            generator_f_optimizer=self.generator_f_optimizer,
            discriminator_x_optimizer=self.discriminator_x_optimizer,
            discriminator_y_optimizer=self.discriminator_y_optimizer
            )
        
        # self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_path, max_to_keep=5)

        self.summary_dir = self.path + "logs/"
        self.chkpoint_dir = self.path + "training_checkpoints/"
        create_dir(self.summary_dir)
        create_dir(self.chkpoint_dir)
        # self.summary_dir = self.log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.summary_dir)
        
        self.summary_step = 1
    
    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.lambda_value * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.lambda_value * 0.5 * loss

    def generate_images(model, test_input):
        prediction = model(test_input)
            
        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
    
    def train_step(self, real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.
            
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            # same_x and same_y are used for identity loss.
            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)
            
            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)
            
            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)
        
        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                                self.generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                                self.generator_f.trainable_variables)
        
        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                    self.discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                    self.discriminator_y.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                                    self.generator_g.trainable_variables))

        self.generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                                    self.generator_f.trainable_variables))
        
        self.discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                        self.discriminator_x.trainable_variables))
        
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                        self.discriminator_y.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('Generator_G_loss', total_gen_g_loss, step=self.summary_step)
            tf.summary.scalar('Generator_F_loss', total_gen_f_loss, step=self.summary_step)
            tf.summary.scalar('Discriminator_X_loss', disc_x_loss, step=self.summary_step)
            tf.summary.scalar('Discriminator_Y_loss', disc_y_loss, step=self.summary_step)
        self.summary_step = self.summary_step + 1
        return total_gen_g_loss, total_gen_f_loss, disc_x_loss,  disc_y_loss
    
    def train(self, dataset, sample_soiled_image, sample_unsoiled_image, chkpoint_step = 1):
        time_list = []
        if self.enable_function:
            self.train_step = tf.function(self.train_step)
        
        for epoch in range(self.epochs):
            start_time = time.time()

            n = 0
            # dataset = tf.data.Dataset.zip((train_horses, train_zebras))
            for image_x, image_y in dataset:
                total_gen_g_loss, total_gen_f_loss, disc_x_loss,  disc_y_loss = self.train_step(image_x, image_y)
                
            wall_time_sec = time.time() - start_time
            time_list.append(wall_time_sec)
            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % chkpoint_step == 0:
                self.checkpoint.save(self.chkpoint_dir)
            template = 'Epoch {}, Generator losses {}, {}, Discriminator Losses {}, {}'
            print (template.format(epoch, total_gen_g_loss, total_gen_f_loss, disc_x_loss,  disc_y_loss))

            display_list1 = self.generate_images(self.generator_g, sample_unsoiled_image)
            display_list2 = self.generate_images(self.generator_f, sample_soiled_image)

            with self.summary_writer.as_default():
                tf.summary.image("Generator G", display_list1, step=epoch)
                tf.summary.image("Generator F", display_list2, step=epoch)

        return time_list

    def generate_images(self, model, test_input, display_fig = False):
        prediction = model(test_input)
        display_list = [test_input[0], prediction[0]]   

        if display_fig:
          plt.figure(figsize=(12, 12))
          title = ['Input Image', 'Predicted Image']

          for i in range(2):
              plt.subplot(1, 2, i+1)
              plt.title(title[i])
              # getting the pixel values between [0, 1] to plot it.
              plt.imshow(display_list[i] * 0.5 + 0.5)
              plt.axis('off')
          plt.show()
        return display_list




