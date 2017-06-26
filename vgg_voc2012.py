import re
from scipy.misc import imread, imresize
#import skimage
#import skimage.io
#import skimage.transform
import numpy as np

import argparse
import sys

#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
#change to vgg_trainable....
import vgg16_trainable as vgg16

FLAGS = None
class_num = 2

voc_jpeg_path = 'VOCdevkit/VOC2012/JPEGImages/'
voc_text_path = 'VOCdevkit/VOC2012/ImageSets/Main/'

def make_flag(flag):
  result = []
  for i in range(class_num):
    result.append(0)
  result[flag] = 1
  return result

def get_set(set_filenames):
  result = []
  sum_count = 0
  for i in range(len(set_filenames)):
    filename = set_filenames[i]
    with open(voc_text_path+set_filenames[i]) as f:
      count = 0
      print("set_filenames %d is %s"%(i, set_filenames[i]))
      lines = f.readlines()
      for line in lines:
        l = line
	#test
	#print(line)
	#test
        line = line.split('\n')[0]
        line = re.split('\s+', line)
        if int(line[1]) == 1:
          count += 1
          result.append({
            'image_name': line[0] + '.jpg',
            'flag': make_flag(i)
          })
      sum_count += count
      print("count is %d"%count)
      print("sum_count is %d"%sum_count)
  return result

def set_resolve(set_arr):
  value_arr = []
  reference_arr = []
  for item in set_arr:
    value_arr.append(get_image(item['image_name']))
    reference_arr.append(item['flag'])
  return value_arr, reference_arr


def get_image(image_name):
  img = imread(voc_jpeg_path + image_name, mode='RGB')

  #test
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
  #resize_img = skimage.transform.resize(crop_img, (224, 224))
  #test

  resize_img = imresize(crop_img, (224, 224))
  return resize_img


train_filenames = [
  'bicycle_train.txt',    'chair_train.txt',        'pottedplant_train.txt',
  'aeroplane_train.txt',  'cat_train.txt',          'person_train.txt',
  'bird_train.txt',       'cow_train.txt',          'sheep_train.txt',
  'boat_train.txt',       'diningtable_train.txt',  'sofa_train.txt',
  'bottle_train.txt',     'dog_train.txt',          'train_train.txt',
  'bus_train.txt',        'horse_train.txt',        'tvmonitor_train.txt',
  'car_train.txt',        'motorbike_train.txt'
]
val_filenames = [
  'bicycle_val.txt',    'chair_val.txt',        'pottedplant_val.txt',
  'aeroplane_val.txt',  'cat_val.txt',          'person_val.txt',
  'bird_val.txt',       'cow_val.txt',          'sheep_val.txt',
  'boat_val.txt',       'diningtable_val.txt',  'sofa_val.txt',
  'bottle_val.txt',     'dog_val.txt',          'train_val.txt',
  'bus_val.txt',        'horse_val.txt',        'tvmonitor_val.txt',
  'car_val.txt',        'motorbike_val.txt'
]

def get_hv():
  train_value = []
  train_reference = []
  for i in range(1000):
    image_path = 'tmp_img/h' + str(i) + '.png'
    img = imread(image_path, mode='RGB')
    img = imresize(img, (224, 224))
    train_value.append(img)
    train_reference.append([1,0])
  for i in range(1000):
    image_path = 'tmp_img/v' + str(i) + '.png'
    img = imread(image_path, mode='RGB')
    img = imresize(img, (224, 224))
    train_value.append(img)
    train_reference.append([0,1])
  return train_value, train_reference  

def main():
  # train_value, train_reference = get_hv()
  # val_value = train_value[:500]
  # val_reference = train_reference[:500]
  # train_value = train_value[500:]
  # train_reference = train_reference[500:]
  #train_set = get_set(train_filenames[:2]) # len = 6540 =  60 * 109
  #val_set = get_set(val_filenames[:2])     # len = 6860

  # my change: '2' to '20'
  train_set = get_set(train_filenames[:class_num]) # len = 6540 =  60 * 109
  val_set = get_set(val_filenames[:class_num])     # len = 6860


  print ("train pic is %d, val_set is %d"%(len(train_set), len(val_set)))
  print 'read image into memory...'
  train_value, train_reference = set_resolve(train_set)
  val_value, val_reference =  set_resolve(val_set)
  
  # train_value = train_value + val_value
  # train_reference = train_reference + val_reference


  # Create the model
  x = tf.placeholder(tf.float32, [None, 224, 224, 3])
  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, class_num])
  
  #train_mode
  train_mode = tf.placeholder(tf.bool)

  # Build the graph for the deep net
  vgg = vgg16.Vgg16('vgg16.npy');
  vgg.build(x, train_mode);
  #vgg.build(x);
  
  y_conv = vgg.prob;
  #raw_soft = tf.nn.softmax(y_conv)

  #cross_entropy = tf.reduce_mean(
      #tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

  #cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
  loss = tf.reduce_sum((y_conv - y_) ** 2)
  #loss = -tf.reduce_sum(y_*tf.log(y_conv))
  #print loss

  #train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
  train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  batch_size = 30 # 60

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
      print "i = ", i
      #print(sess.run(loss))
      #print (sess.run(cross_entropy))
      for j in range(len(train_value)/batch_size - 1):
        #train_step.run(feed_dict={x: train_value[j*batch_size:(j+1)*batch_size],
                        #y_: train_reference[j*batch_size:(j+1)*batch_size], train_mode: True})
	sess.run(train_step, feed_dict={x: train_value[j*batch_size:(j+1)*batch_size],
                        y_: train_reference[j*batch_size:(j+1)*batch_size], train_mode: True})
      curr_loss = sess.run([loss], feed_dict={x: train_value[j*batch_size:(j+1)*batch_size],
                        y_: train_reference[j*batch_size:(j+1)*batch_size], train_mode: False})
      print("loss: %s"%(curr_loss))
	#print loss
      #vgg.save_npy("/weight_data/vgg16-save" + str(i) + ".npy")
      #train_mode: True
      
      if i % 1 == 0:
        sum_acc = 0.0
        for k in range(len(train_value)/batch_size - 1):
          sum_acc += accuracy.eval(feed_dict={
              x: train_value[k*batch_size:(k+1)*batch_size],
              y_: train_reference[k*batch_size:(k+1)*batch_size], train_mode: False})
	  print('k is %d, sum_acc is %g' % (k, sum_acc))
        sum_acc /= len(train_value)/batch_size - 1
	#print('step %d, loss %g' % (i, loss[0]))
	#print loss.eval()
        print('step %d, training accuracy %g' % (i, sum_acc))

    	sum_acc = 0.0
    	for j in range(len(val_value)/batch_size - 1):
      	    sum_acc += accuracy.eval(feed_dict={
              x: val_value[j*batch_size:(j+1)*batch_size],
              y_: val_reference[j*batch_size:(j+1)*batch_size], train_mode: False})
    	sum_acc /= len(val_value)/batch_size - 1
    	print('test accuracy %g' % sum_acc)

if __name__ == '__main__':
  main()
