import gzip
import numpy


def read_data_sets(data_type="train"):
    """
    Parse or download mnist data if train_dir is empty.

    :param data_type: Reading training set or testing set.
           It can be either "train" or "test"
    :return: (ndarray, ndarray) representing (features, labels)
    """
    TRAIN_DATA = './Census/train_tensor.data'
    TRAIN_LABELS = './Census/train_label.data'
    TEST_DATA = './Census/test_tensor.data'
    TEST_LABELS = './Census/test_label.data'

    if data_type == "train":
        data = numpy.loadtxt(TRAIN_DATA, delimiter=',')
        labels = numpy.loadtxt(TRAIN_LABELS)
        return data, labels
    else:
        data = numpy.loadtxt(TEST_DATA, delimiter=',')
        labels = numpy.loadtxt(TEST_LABELS)
        return data, labels


def get_data(data_type='train'):
    features, labels = read_data_sets(data_type)
    features = features.astype('int64')
    return features, labels


if __name__ == "__main__":
    train_images, train_labels = get_data("train")
    print(train_images.shape)
    print(train_labels.shape)
