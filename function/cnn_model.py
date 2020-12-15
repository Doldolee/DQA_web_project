import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from tqdm import tqdm

# 전처리한 데이터 불러오기
DATA_IN_PATH='./'
DATA_OUT_PATH = './cnn_data'
INPUT_TRAIN_DATA='nsmc_train_input.npy'
LABEL_TRAIN_DATA='nsmc_train_label.npy'
DATA_CONFIGS = 'data_configs.json'

train_input = np.load(open(DATA_IN_PATH + INPUT_TRAIN_DATA, 'rb'))
train_input = pad_sequences(train_input, maxlen=train_input.shape[1])
train_label = np.load(open(DATA_IN_PATH + LABEL_TRAIN_DATA, 'rb'))
prepro_configs = json.load(open(DATA_IN_PATH+DATA_CONFIGS,'r'))

# 모델에 필요한 파라미터 정의
model_name = 'cnn_classifier_kr'
BATCH_SIZE=512
NUM_EPOCHS = 2
VALID_SPLIT=0.1
MAX_LEN = train_input.shape[1]

kargs = {'model_name':model_name,
          'vocab_size':prepro_configs['vocab_size'],
          'embedding_size':128,
          'num_filters':100,
          'fropout_rate':0.5,
          'hidden_dimension':250,
          'output_dimension':1}

# 모델함수
class CNNClassifier(tf.keras.Model):

  def __init__(self, **kargs):
    super(CNNClassifier, self).__init__(name=kargs['model_name'])
    self.embedding = layers.Embedding(input_dim=kargs['vocab_size'],
                                    output_dim = kargs['embedding_size'])
    self.conv_list = [layers.Conv1D(filters=kargs['num_filters'],
                                    kernel_size=kernel_size,
                                    padding='valid',
                                    activation=tf.keras.activations.relu,
    kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
      for kernel_size in [3,4,5]]
    self.pooling = layers.GlobalMaxPooling1D()
    self.dropout = layers.Dropout(kargs['dropout_rate'])
    self.fc1 = layers.Dense(units=kargs['hidden_dimension'],
        activation=tf.keras.activations.relu,
        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
    self.fc2 = layers.Dense(units=kargs['output_dimension'],
        activation=tf.keras.activations.sigmoid,
        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))

  def call(self, x):
    x=self.embedding(x)
    x=self.dropout(x)
    
    x = self.fc1(x)
    x = self.fc2(x)

    return x


# 모델 학습
model = CNNClassifier(**kargs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])


earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=2)

checkpoint_path = DATA_OUT_PATH+model_name + '/weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

if os.path.exists(checkpoint_dir):
  print("{}--Folder already exists \n".format(checkpoint_dir))
else:
  os.makedirs(checkpoint_dir, exist_ok=True)
  print("{}--Folder create complete \n".format(checkpoint_dir))

cp_callback = ModelCheckpoint(
  checkpoint_path, monitor = 'val_accuracy', verbose=1, save_best_only=True,
  save_weights_only=True)

history = model.fit(train_input, train_label, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])


INPUT_TEST_DATA = 'nsmc_test_input.npy'
LABEL_TEST_DATA = 'nsmc_test_label.npy'
SAVE_FILE_NM = 'weights.h5'

test_input = np.load(open(DATA_IN_PATH + INPUT_TEST_DATA, 'rb'))
test_input = pad_sequences(test_input, maxlen=test_input.shape[1])
test_label_data = np.load(open(DATA_IN_PATH + LABEL_TEST_DATA, 'rb'))

model.load_weights(os.path.join(DATA_OUT_PATH, model_name, SAVE_FILE_NM))
print(model.evaluate(test_input, test_label_data))
print('x')

