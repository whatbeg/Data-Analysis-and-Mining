import gzip
import numpy

TRAIN_MEAN = 0.13066047740239506
TRAIN_STD = 0.3081078
TEST_MEAN = 0.13251460696903547
TEST_STD = 0.31048024


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """
    Extract the images into a 4D uint8 numpy array [index, y, x, depth]

    :param f: Data file object to be read by gzip
    :return: A 4D unit8 numpy array [index, y, x, depth]
    :raise ValueError: If the bytestream doesn't start with 2051.
    """

    # print('Extracting {}'.format(f.name))
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, 1, rows, cols)
        return data


def extract_labels(f):
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels


def read_data_sets(data_type="train"):
    """
    Parse or download mnist data if train_dir is empty.

    :param data_type: Reading training set or testing set.
           It can be either "train" or "test"
    :return: (ndarray, ndarray) representing (features, labels)
    """
    TRAIN_IMAGES = './mnist/train-images-idx3-ubyte.gz'
    TRAIN_LABELS = './mnist/train-labels-idx1-ubyte.gz'
    TEST_IMAGES = './mnist/t10k-images-idx3-ubyte.gz'
    TEST_LABELS = './mnist/t10k-labels-idx1-ubyte.gz'

    if data_type == "train":
        with open(TRAIN_IMAGES, 'rb') as f:
            train_images = extract_images(f)
        with open(TRAIN_LABELS, 'rb') as f:
            train_labels = extract_labels(f)
        return train_images, train_labels
    else:
        with open(TEST_IMAGES, 'rb') as f:
            test_images = extract_images(f)
        with open(TEST_LABELS, 'rb') as f:
            test_labels = extract_labels(f)
        return test_images, test_labels


def Normalize(tensor, mean, std):
    """
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel to `channel = (channel - mean) / std`
    Given mean: GreyLevel and std: GrayLevel,
    will normalize the channel to `channel = (channel - mean) / std`

    :param tensor: image tensor to be normalized
    :param mean: mean of every channel
    :param std: standard variance of every channel
    :return: normalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t -= m
        t /= s
    return tensor


def get_data(data_type='train'):
    images, labels = read_data_sets(data_type)
    images = images.astype('float32')
    images /= 255
    return images, labels


if __name__ == "__main__":
    train_images, train_labels = get_data("train")
    # for (data, label) in zip(train_images, train_labels):
    #     data = Normalize(data, (TRAIN_MEAN, ), (TRAIN_STD, ))
    #     print(data)
    #     print(label)
