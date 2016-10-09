'''
This scripts creats list of image's path and transform images of imagnet into 
TFRecord file.
'''
import tensorflow as tf
import os
import cv2
from scipy import misc
ImageNetDir = '/media/1T/ImageNet/raw_jpg/'
def create_images_path_list_and_labels(results_path):
    classes = os.listdir(ImageNetDir)
    images_list = open(results_path, 'w')
    for idx, class_dir in enumerate(classes):
        images = os.listdir(os.path.join(ImageNetDir, class_dir))
        for img in images:
            images_list.write(str(idx)+','+class_dir+','+img+'\n')
    print len(classes)
def create_tfrecords(images_list, records_path, resize = (224, 224)):
    writer = tf.python_io.TFRecordWriter(records_path)
    num_example = 0
    with open(images_list, 'r') as f:
        for l in f.readlines():
            print num_example
            try:
                l = l.split(',')
                image = misc.imread(os.path.join(ImageNetDir, l[1], l[2].strip()))
                if resize is not None:
                    image = misc.imresize(image, resize)
                height, width, channel = image.shape
                label = int(l[0])
                example = tf.train.Example(features = tf.train.Features(feature = {
                    'height': tf.train.Feature(int64_list = tf.train.Int64List(value = [height])),
                    'width': tf.train.Feature(int64_list = tf.train.Int64List(value = [width])),
                    'channel': tf.train.Feature(int64_list = tf.train.Int64List(value = [channel])),
                    'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image.tobytes()])),
                    'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
                    }))
                serialized = example.SerializeToString()
                writer.write(serialized)
                num_example += 1
            except:
                continue
        writer.close()
if __name__ == '__main__':
    #create_images_path_list_and_labels('train.txt')
    create_tfrecords('train.txt', '/media/1T/ImageNet/train.record')

