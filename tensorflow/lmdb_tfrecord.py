import numpy as np
import cv2


import lmdb
import os
import tensorflow as tf

def generate_images_in_folders():

    # Create folders if they don't exist
    os.makedirs('train_folder', exist_ok=True)
    os.makedirs('ground_truth_folder', exist_ok=True)

    for i in range(4):
        # Generate a random clear image (for example, 128x128 RGB)
        clear_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

        # Create a blurred version using Gaussian blur
        blurred_img = cv2.GaussianBlur(clear_img, (11, 11), 0)

        # Save images
        cv2.imwrite(f'train_folder/image_{i + 1}.png', blurred_img)
        cv2.imwrite(f'ground_truth_folder/image_{i + 1}.png', clear_img)

    print("Image pairs generated and saved.")

def generate_tfrecord():
    train_folder = 'train_folder'
    ground_folder = 'ground_truth_folder'
    output_tfrecord = 'dataset.tfrecord'

    image_filenames = [f for f in os.listdir(train_folder) if f.endswith('.png')]

    with tf.io.TFRecordWriter(output_tfrecord) as writer:
        for filename in image_filenames:
            train_path = os.path.join(train_folder, filename)
            train_image = tf.io.read_file(train_path)
            train_image = tf.image.decode_png(train_image, channels=3)
            train_png = tf.io.encode_png(train_image)

            ground_path = os.path.join(ground_folder, filename)
            ground_image = tf.io.read_file(ground_path)
            ground_image = tf.image.decode_png(ground_image, channels=3)
            ground_png = tf.io.encode_png(ground_image)

            feature = {
                'train_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_png.numpy()])),
                'ground_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ground_png.numpy()])),
                'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()]))
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    print(f'TFRecord file created: {output_tfrecord}')

def read_tfrecord():
    feature_description = {
        'train_image': tf.io.FixedLenFeature([], tf.string),
        'ground_image': tf.io.FixedLenFeature([], tf.string),
        'filename': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        train_image = tf.image.decode_png(parsed['train_image'], channels=3)
        ground_image = tf.image.decode_png(parsed['ground_image'], channels=3)
        filename = parsed['filename']
        return train_image, ground_image, filename

    tfrecord_path = 'dataset.tfrecord'
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    for train_img, ground_img, fname in parsed_dataset:
        print("Filename:", fname.numpy().decode())
        print("Train image shape:", train_img.shape)
        print("Ground image shape:", ground_img.shape)


def generate_lmdb():


    train_folder = 'train_folder'
    ground_folder = 'ground_truth_folder'
    output_lmdb = 'dataset.lmdb'

    image_filenames = [f for f in os.listdir(train_folder) if f.endswith('.png')]

    env = lmdb.open(output_lmdb, map_size=1<<30)
    with env.begin(write=True) as txn:
        for filename in image_filenames:
            train_path = os.path.join(train_folder, filename)
            ground_path = os.path.join(ground_folder, filename)

            with open(train_path, 'rb') as f:
                train_bytes = f.read()
            with open(ground_path, 'rb') as f:
                ground_bytes = f.read()

            # Store both images as a tuple, separated by a special marker if needed
            key = filename.encode()
            value = train_bytes + b'|||' + ground_bytes
            txn.put(key, value)

    print(f'LMDB file created: {output_lmdb}')


def read_lmdb():
    def lmdb_generator(lmdb_path):
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                # Split the value into train and ground images
                train_bytes, ground_bytes = value.split(b'|||')
                train_image = tf.image.decode_png(train_bytes, channels=3)
                ground_image = tf.image.decode_png(ground_bytes, channels=3)
                filename = key.decode()
                yield train_image, ground_image, filename

    lmdb_path = 'dataset.lmdb'
    output_types = (tf.uint8, tf.uint8, tf.string)
    output_shapes = (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3]), tf.TensorShape([]))

    lmdb_dataset = tf.data.Dataset.from_generator(
        lambda: lmdb_generator(lmdb_path),
        output_types=output_types,
        output_shapes=output_shapes
    )

    for train_img, ground_img, fname in lmdb_dataset:
        print("Filename:", fname.numpy())
        print("Train image shape:", train_img.shape)
        print("Ground image shape:", ground_img.shape)
if __name__ == '__main__':
    # generate_images_in_folders()
    # generate_tfrecord()
    # read_tfrecord()
   # generate_lmdb()
    read_lmdb()