import numpy as np
import matplotlib.pyplot as plt
import argparse


def analyse_pytorch(files, title):
    """
    analyse pytorch like log.
    For example,
    `Train Epoch: 1 [20480/32561 (62%)]	Loss: 0.598629`
    `Test set: Average loss: 0.0024, Accuracy: 12302/16281 (75.6%)`

    :param files: log file list
    :param title: figures title
    :return: None
    """
    assert len(files) > 0
    for filename in files:
        testloss, top1acc = [], []
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line.count('Test set: Average loss'):
                    testloss.append(float(line.strip().split(' ')[6][:-1]))
                    top1acc.append(float(line.strip().split(' ')[9][1:-2]))

        plt.figure(1)
        plt.title(title)
        plt.ylabel("Top1 Accuracy (%)")
        plt.xlabel("Epoch")
        plt.plot(range(1, len(top1acc) + 1), top1acc, label=filename[:-4])
        plt.legend(loc="lower right")
        plt.grid()
        plt.figure(2)
        plt.title(title)
        plt.ylabel("Test Loss")
        plt.xlabel("Epoch")
        plt.plot(range(1, len(testloss) + 1), testloss, label=filename[:-4])
        plt.legend(loc="upper right")
        plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="log file name"
    )
    FLAGS, unparsed = parser.parse_known_args()
    # print(FLAGS.file)
    analyse_pytorch([FLAGS.file, ], "{}".format(FLAGS.file[:-4]))
