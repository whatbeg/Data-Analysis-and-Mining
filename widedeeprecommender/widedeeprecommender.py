#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Still in experimental stage!

from optparse import OptionParser
import sys
import pandas as pd
import numpy as np
# from ...dataset.transformer import *
# from ...nn.layer import *
# from ...nn.criterion import *
# from ...optim.optimizer import *
# from ...util.common import *

from dataset.transformer import *
from nn.layer import *
from nn.criterion import *
from optim.optimizer import *
from util.common import *

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


def build_models(model_type='wide_n_deep', classNum=2):

    model = Sequential()
    wide_model = Concat(2)
    for i in range(1, 8):
        wide_model.add(Sequential().add(Select(2, i)).add(Reshape([1])))
    deep_model = Sequential()
    deep_column = Concat(2)
    deep_column.add(Sequential().add(Select(2, 11)).add(LookupTable(9, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 12)).add(LookupTable(16, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 13)).add(LookupTable(2, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 14)).add(LookupTable(6, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 15)).add(LookupTable(42, 8, 0.0)))
    deep_column.add(Sequential().add(Select(2, 16)).add(LookupTable(15, 8, 0.0)))
    for i in range(17, 22):
        deep_column.add(Sequential().add(Select(2, i)).add(Reshape([1])))
    deep_model.add(deep_column).add(Linear(53, 100)).add(ReLU()).add(Linear(100, 50)).add(ReLU())
    if model_type == 'wide_n_deep':
        wide_model.add(deep_model)
        model.add(wide_model).add(Linear(57, classNum)).add(LogSoftMax())
        return model
    elif model_type == 'wide':
        model.add(wide_model).add(Linear(7, classNum)).add(LogSoftMax())
        return model
    elif model_type == 'deep':
        model.add(deep_model).add(Linear(50, classNum)).add(LogSoftMax())
        return model
    else:
        raise ValueError("Not valid model type. Only for wide, deep, wide_n_deep!")


def get_data_rdd(sc, data_type='train'):

    if data_type == 'train':
        data_tensor = './census/train_tensor.data'
        data_label = './census/train_label.data'
    elif data_type == 'test':
        data_tensor = './census/test_tensor.data'
        data_label = './census/test_label.data'
    else:
        raise ValueError("Not valid Data Type, only 'train' or 'test' !")
    features = np.loadtxt(data_tensor, delimiter=',')
    labels = np.loadtxt(data_label)
    features = sc.parallelize(features)
    labels = sc.parallelize(labels)
    record = features.zip(labels).map(lambda features_label:
                                      Sample.from_ndarray(features_label[0], features_label[1]+1))
    return record


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="128")
    parser.add_option("-m", "--model", dest="model_type", default="wide_n_deep")
    (options, args) = parser.parse_args(sys.argv)

    sc = SparkContext(appName="wide_n_deep", conf=create_spark_conf())
    init_engine()

    if options.action == "train":
        train_data = get_data_rdd(sc, 'train')
        test_data = get_data_rdd(sc, 'test')
        state = {"learningRate": 0.001,
                 "learningRateDecay": 0.0005}
        optimizer = Optimizer(
            model=build_models(options.model_type, 2),
            training_rdd=train_data,
            criterion=ClassNLLCriterion(),
            optim_method="Adam",
            state=state,
            end_trigger=MaxEpoch(20),
            batch_size=int(options.batchSize))
        optimizer.set_validation(
            batch_size=256,
            val_rdd=test_data,
            trigger=EveryEpoch(),
            val_method=["Top1Accuracy", "Loss"]
        )

        optimizer.set_checkpoint(EveryEpoch(), "/tmp/{}/".format(options.model_type))
        trained_model = optimizer.optimize()
        parameters = trained_model.parameters()
        results = trained_model.test(test_data, 256, ["Top1Accuracy"])
        for result in results:
            print(result)
    elif options.action == "test":
        # Load a pre-trained model and then validate it through top1 accuracy.
        test_data = get_data_rdd(sc, 'test')
        # TODO: Pass model path through external parameter
        model = Model.load("/tmp/{}/model.5101".format(options.model_type))
        results = model.test(test_data, 256, ["Top1Accuracy"])
        for result in results:
            print(result)
