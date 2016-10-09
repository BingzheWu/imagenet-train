import tensorflow as tf
def decode_from_tfrecords( filename, num_epoch = None):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    example = tf.parse_single_example(serialized, features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channel': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
        })
    #example = tf.parse_single_example(serialized, dense_keys = ['image', 'width', 'height', 'label', 'channel'], dense_types = [tf.string, tf.int64, tf.int64, tf.int64, tf.int64])
    print example['height'].get_shape()
    label = tf.cast(example['label'], tf.int32)
    image = tf.decode_raw(example['image'], tf.uint8)
    print image.get_shape().as_list()
    image = tf.reshape(image, tf.pack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['channel'], tf.int32)
        ]))
    #image = tf.reshape(image, [224, 224, 3])
    image.set_shape([224,224,3])
    return image, label
def get_train_batch(image, label, batch_size, crop_size = None):
    if crop_size:
        distorted_image = tf.random_crop(image, [crop_size, crop_size,3])
    distorted_image = tf.image.random_flip_up_down(image)
    images, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size = batch_size, num_threads = 16, capacity = 50000, min_after_dequeue = 10000)
    return images, tf.reshape(label_batch, [batch_size])

if __name__ == '__main__':
    image, label = decode_from_tfrecords('/media/1T/ImageNet/train1.record', num_epoch = 1)
    batch_image, label = get_train_batch(image, label, batch_size = 50)
    sess = tf.Session()
    #threads = tf.train.start_queue_runners(sess = sess)
    init = tf.initialize_all_variables()
    sess.run(init)
    #val , l = sess.run([batch_image, label])
    #print(val.shape,l)
