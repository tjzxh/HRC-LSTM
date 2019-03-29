import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from Retrain_Data import retrain
from keras.callbacks import TensorBoard
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras import backend as K

K.set_session(sess)
# all_useful_data = np.loadtxt(open(r"C:\Users\29904\Desktop\HRC LSTM\final_data0.csv", "rb"), delimiter=",", skiprows=0)
all_pd = pd.read_csv(r"C:\Users\29904\Desktop\HRC LSTM\final_data0.csv", delimiter=",", skiprows=0)
all_useful_data = all_pd.to_numpy()
veh_id = all_useful_data[:, 0]
all_useful_data[:, 2], all_useful_data[:, 4], all_useful_data[:, 6], all_useful_data[:, 8], all_useful_data[:,
                                                                                            10], all_useful_data[:,
                                                                                                 12], all_useful_data[:,
                                                                                                      14] = all_useful_data[
                                                                                                            :,
                                                                                                            2] / 65, all_useful_data[
                                                                                                                     :,
                                                                                                                     4] / 65, all_useful_data[
                                                                                                                              :,
                                                                                                                              6] / 65, all_useful_data[
                                                                                                                                       :,
                                                                                                                                       8] / 65, all_useful_data[
                                                                                                                                                :,
                                                                                                                                                10] / 65, all_useful_data[
                                                                                                                                                          :,
                                                                                                                                                          12] / 65, all_useful_data[
                                                                                                                                                                    :,
                                                                                                                                                                    14] / 65
all_useful_data[:, 3], all_useful_data[:, 5], all_useful_data[:, 7], all_useful_data[:, 9], all_useful_data[:,
                                                                                            11], all_useful_data[:,
                                                                                                 13], all_useful_data[:,
                                                                                                      15] = all_useful_data[
                                                                                                            :,
                                                                                                            3] / 1650, all_useful_data[
                                                                                                                       :,
                                                                                                                       5] / 1650, all_useful_data[
                                                                                                                                  :,
                                                                                                                                  7] / 1650, all_useful_data[
                                                                                                                                             :,
                                                                                                                                             9] / 1650, all_useful_data[
                                                                                                                                                        :,
                                                                                                                                                        11] / 1650, all_useful_data[
                                                                                                                                                                    :,
                                                                                                                                                                    13] / 1650, all_useful_data[
                                                                                                                                                                                :,
                                                                                                                                                                                15] / 1650
short_veh_id = list(set(list(veh_id)))
short_veh_id.sort()
all_input = []
print("Experiment starts")
for i in range(int(all_useful_data.shape[0] * 0.75) - 2 * 80):
    if all_useful_data[i, 0] == all_useful_data[i + 79, 0] and all_useful_data[i + 79, 1] - all_useful_data[
        i, 1] == 79 and all_useful_data[i + 79, 0] == all_useful_data[i + 158, 0] and all_useful_data[i + 158, 1] - \
            all_useful_data[i + 79, 1] == 79:
        no_use = all_useful_data[i + 80:i + 2 * 80]
        no_use = np.array(no_use)
        the_output = no_use[:, :]
        all_together = np.hstack((all_useful_data[i: i + 80][:, :], the_output))
        all_input.append(all_together)
print("Data processing over")
all_input = np.array(all_input)
row0 = round(0.5 * all_input.shape[0])
row1 = round(1 * all_input.shape[0])
train = all_input[:row0, :, 2:]
# np.random.shuffle(train)
x_train = train[:, :, :14]
y_train = train[:, 0, 16:18]

x_test_id = all_input[row0:, :, 0:2]
x_test = all_input[row0:, :, 2:16]
y_test_id = all_input[row0:, 0, 16:18]
y_test = all_input[row0:, 0, 18:20]
y_test_environ = all_input[row0:, 0, 20:]
# print(x_test)

# x_valid_id = all_input[row1:, :, 0:2]
# x_valid = all_input[row1:, :, 2:16]
# y_valid_id = all_input[row1:, 0, 16:18]
# y_valid = all_input[row1:, 0, 18:20]
# y_valid_environ = all_input[row1:, 0, 20:]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 14))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 14))
# x_valid = np.reshape(x_valid, (x_valid.shape[0], x_valid.shape[1], 14))

y_train = np.reshape(y_train, (y_train.shape[0], 2))
y_test = np.reshape(y_test, (y_test.shape[0], 2))
# y_valid = np.reshape(y_valid, (y_valid.shape[0], 2))

model = Sequential()
# Stack LSTM
model.add(LSTM(input_shape=(None, 14), units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=2))
model.add(Activation("sigmoid"))
# start = time.time()
model.compile(loss="mse", optimizer="adam", metrics=['acc'])
early_stopping = EarlyStopping(monitor='val_loss', patience=0)
iniCallBack = TensorBoard(log_dir='logs/initrain{}'.format(0),  # log 目录
                          update_freq=12000)
history = model.fit(x_train, y_train, batch_size=32, epochs=15, validation_split=0.05,
                    callbacks=[early_stopping, iniCallBack], shuffle=True)

num = 0
while num < 5:
    tbCallBack = TensorBoard(log_dir='logs/retrain{}'.format(num),  # log 目录
                             update_freq=12000)
    all_re_input, no_use_predict, cons_prop = retrain(x_test_id, x_test, model)
    all_re_input = np.array(all_re_input)
    fianl_input = np.vstack((x_train, all_re_input))
    final_input = np.reshape(fianl_input, (fianl_input.shape[0], fianl_input.shape[1], 14))
    final_output = np.vstack((y_train, y_test))
    history = model.fit(final_input, final_output, batch_size=32, epochs=5, validation_split=0.05,
                        callbacks=[early_stopping, tbCallBack], shuffle=True)
    num += 1

model.save('my-model-2019-03-27.h5')
print("Finish training!")
