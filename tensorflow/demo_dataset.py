import tensorflow as tf
import numpy as np

def demo_inmemory_dataset():
    features = np.array([1, 2, 3, 4, 5])
    labels = np.array([0, 1, 0, 1, 0])
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    for feature, label in dataset:
        print(f"Feature: {feature.numpy()}, Label: {label.numpy()}")

def demo_from_generator():
    def simple_generator():
         for i in range(10):
             yield i

    dataset = tf.data.Dataset.from_generator(
    simple_generator,
    output_signature = tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    for data in dataset:
        print(data.numpy())

def demo_from_callable_class():
    class InfiniteGenerator:
        def __init__(self, array):
            self.array = array
            self.index = 0

        def __call__(self):
            while True:
                yield self.array[self.index]
                self.index = (self.index + 1) % len(self.array)

    generator_obj = InfiniteGenerator([1, 5, 9, 21])

    dataset = tf.data.Dataset.from_generator(
        generator_obj,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    # Since this is an infinite dataset, we use .take() to get a finite number of elements.
    for data in dataset.take(9):
        print(data.numpy())

def demo_dataset_shape():
    class ImageGenerator:
        def __call__(self):
            for i in range(10):
                # Yield a random 64x64 RGB image and a label.
                yield tf.random.normal(shape=[64, 64, 3]), i

    # 1. Create the dataset from a generator.
    dataset = tf.data.Dataset.from_generator(
        ImageGenerator(),
        output_signature=(
            tf.TensorSpec(shape=[64, 64, 3], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    # 2. Shuffle the data to ensure the model sees it in a random order.
    dataset = dataset.shuffle(buffer_size=10)

    # 3. Batch the data. This groups elements into batches.
    dataset = dataset.batch(4)

    # 4. Prefetch data for performance.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Now, let's inspect the shape of an element from the final dataset.
    for images, labels in dataset.take(1):
        print(f"Shape of images batch: {images.shape}")
        print(f"Shape of labels batch: {labels.shape}")

def demo_iterable_dataset1():
    class MyObject(object):
        def __init__(self, start, end):
            self.current = start
            self.end = end

        def __iter__(self):
            return self

        def __next__(self):
            if self.current < self.end:
                result = self.current
                self.current += 1
                return result
            else:
                raise StopIteration

    data_generator = MyObject(1, 10)
    data = tf.data.Dataset.from_generator(lambda: data_generator,
                                          output_signature=
                                          tf.TensorSpec(shape=(), dtype=tf.int32))
    for d in data:
        print(d.numpy())

def demo_iterable_dataset2():
    import tensorflow as tf
    class MyObject(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __getitem__(self, index):
            return list(range(self.start, self.end))[index]

        def __len__(self):
            return self.end - self.start

    data_generator = MyObject(1, 10)
    data = tf.data.Dataset.from_generator(lambda: data_generator,
                                          output_signature=
                                          tf.TensorSpec(shape=(), dtype=tf.int32))
    for d in data:
        print(d.numpy())

def demo_output_dict():
    class MyObject(object):
        def __init__(self, start, end):
            self.current = start
            self.end = end

        def __call__(self, num):
            while self.current < self.end + num:
                yield {'var1': self.current, 'var2': self.current}
                self.current += 1

    data_generator = MyObject(1, 3)
    data_num = 3
    data = tf.data.Dataset.from_generator(data_generator,
                                          args=[data_num, ],
                                          output_signature=(
                                              {
                                                  'var1': tf.TensorSpec(shape=(), dtype=tf.int32),
                                                  'var2': tf.TensorSpec(shape=(), dtype=tf.int32),
                                              }
                                          ))
    for d in data:
        print((d['var1'].numpy(), d['var2'].numpy()))


def demo_output_tuple():
    import tensorflow as tf
    class MyObject(object):
        def __init__(self, start, end):
            self.current = start
            self.end = end

        def __call__(self, num):
            while self.current < self.end + num:
                yield tf.random.normal(shape=[2, 2, 3]), tf.random.normal(shape=[2, 2, 3])
                self.current += 1

    data_generator = MyObject(1, 4)
    data_num = 3
    data = tf.data.Dataset.from_generator(data_generator,
                                          args=[data_num, ],
                                          output_signature=(
                                              tf.TensorSpec(shape=([None, None, None]), dtype=tf.float32),
                                              tf.TensorSpec(shape=([None, None, None]), dtype=tf.float32)), )
    print('before batch operation')
    for d in data:
        print(f"d0: {d[0].shape}, d1: {d[1].shape}")
    print('after batch operation')

    data_generator = MyObject(1, 4)
    data2 = tf.data.Dataset.from_generator(data_generator,
                                           args=[data_num, ],
                                           output_signature=(
                                               tf.TensorSpec(shape=([None, None, None]), dtype=tf.float32),
                                               tf.TensorSpec(shape=([None, None, None]), dtype=tf.float32)), )
    data2 = data2.batch(2)
    for d in data2:
        print(f"d0: {d[0].shape}, d1: {d[1].shape}")

if __name__ == "__main__":
    demo_output_tuple()
    # before batch operation
    # d0: (2, 2, 3), d1: (2, 2, 3)
    # d0: (2, 2, 3), d1: (2, 2, 3)
    # d0: (2, 2, 3), d1: (2, 2, 3)
    # d0: (2, 2, 3), d1: (2, 2, 3)
    # d0: (2, 2, 3), d1: (2, 2, 3)
    # d0: (2, 2, 3), d1: (2, 2, 3)
    # after batch operation
    # d0: (2, 2, 2, 3), d1: (2, 2, 2, 3)
    # d0: (2, 2, 2, 3), d1: (2, 2, 2, 3)
    # d0: (2, 2, 2, 3), d1: (2, 2, 2, 3)
