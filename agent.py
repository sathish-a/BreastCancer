import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
import keras.callbacks as ct

dataframe = pd.read_csv("datasets.data")
dataframe = dataframe.drop('id', 1)

# id,clump_thickness,uniform_cell_size,uniform_cell_shape,marginal_adhesion,single_epi_cell_size,bare_nuclei,bland_chromation,normal_nucleoli,mitoses,class
label1 = dataframe['class']
label = []

for lab in label1:
    if lab == 2:
        label.append([1, 0])  # class 2
    elif lab == 4:
        label.append([0, 1])  # class 4

dataframe = dataframe.drop('class', 1)

dataframe.replace('?',-99999, inplace=True)
data = np.array(dataframe)
label = np.array(label)


model = Sequential()
model.add(Dense(48, input_dim=9, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(2, activation='softmax'))
tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/tflearn_logs/', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(data, label, epochs=1000, batch_size=70, validation_data=(data, label), callbacks=[tbCallBack])
score = model.evaluate(data, label, batch_size=100)

print(score)


test = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
print(model.predict(test))


#0.93991416360175661 accuracy
#loss 0.078175581895742294