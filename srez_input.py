import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def setup_inputs(sess, filenames, image_size=None, capacity_factor=3):

    if image_size is None:
        image_size = FLAGS.sample_size

    # Read each JPEG file
    reader = tf.WholeFileReader()

    featureFilenames = filenames[0]
    labelFilenames = filenames[1]


    filename_queue_features = tf.train.string_input_producer(featureFilenames)
    filename_queue_labels = tf.train.string_input_producer(labelFilenames)

    key, value = reader.read(filename_queue_features)
    channels = 3
    original_image = tf.image.decode_png(value, channels=channels, name="dataset_image")
    original_image.set_shape([300, 300, channels])

    key, value = reader.read(filename_queue_labels)
    channels = 3
    blur_image = tf.image.decode_png(value, channels=channels, name="dataset_image")
    blur_image.set_shape([300, 300, channels])

    feature = original_image
    label   = blur_image

    # Using asynchronous queues
    features, labels = tf.train.batch([feature, label],
                                      batch_size=FLAGS.batch_size,
                                      num_threads=4,
                                      capacity = capacity_factor*FLAGS.batch_size,
                                      name='labels_and_features')

    tf.train.start_queue_runners(sess=sess)

    return features, labels
