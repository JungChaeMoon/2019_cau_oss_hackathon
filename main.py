from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

# tensorflow와 tf.keras 및 관련 라이브러리 임포트
import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

is_mnist = False

# 데이터셋 로드
# x_train, y_train: 트레이닝 데이터 및 레이블
# x_test, y_test: 테스트 데이터 및 레이블
if is_mnist:
  data_type = 'mnist'
  (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data() # fashion MNIST 데이터셋인 경우,
else:
  data_type = 'cifar10'
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data() # cifar10 데이터셋인 경우,


# 분류를 위해 클래스 벡터를 바이너리 매트릭스로 변환
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 총 클래스 개수
num_classes = y_test.shape[1]

# 인풋 데이터 타입
input_shape = x_test.shape[1:]

#데이터 전처리 (예: normalization)
x_train_after = x_train / 255.0
x_test_after = x_test / 255.0

#reverse 'to_categorical'
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

# 인풋 데이터 타입
input_shape = x_train_after.shape[1:]

if is_mnist:
    x_train_after = x_train_after.reshape(x_train_after.shape[0], 28, 28, 1)
    x_test_after = x_test_after.reshape(x_test_after.shape[0], 28, 28, 1)
    input_shape = x_test_after.shape[1:]

filepath = 'content/model'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True ,mode='max')
callbacks_list = [checkpoint]

# 순차 모델 생성 (가장 기본구조)
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

batch_size = 512

gen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    shear_range=0.3,
    height_shift_range=0.08,
    zoom_range=0.08
)
batches = gen.flow(x_train_after, y_train, batch_size=batch_size)
val_batches = gen.flow(x_test_after, y_test, batch_size=batch_size)


model.fit_generator(
    batches,
    steps_per_epoch=48000//batch_size,
    epochs=50,
    validation_data=val_batches,
    validation_steps=12000//batch_size,
    use_multiprocessing=True,
    callbacks=callbacks_list,
)

model.evaluate(x_test_after, y_test)

save_path = '/content/'
team_name = 'team03'

# 모델의 weight 값만 저장합니다.
model.save_weights(save_path + 'model_weight_' + data_type + '_' + team_name + '.h5')

# 모델의 구조만을 저장합니다.
model_json = model.to_json()
with open(save_path + 'model_structure_' + data_type + '_' + team_name + '.json', 'w') as json_file :
    json_file.write(model_json)

# 트레이닝된 전체 모델을 저장합니다.
model.save(save_path +  'model_entire_' + data_type + '_' + team_name + '.h5')

save_path = '/content/'
team_name = 'team03'

model = keras.models.load_model(save_path + 'model_entire_' + data_type + '_' + team_name + '.h5')
model.evaluate(x_test_after, y_test)