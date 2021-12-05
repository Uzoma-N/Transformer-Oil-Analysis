"""
Transformer Oil analysis

using the data given in the excel sheet, analyse the transformer based on it's gaseous contents

Author: Nwiwu Uzoma
"""

import numpy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import read_excel
from gekko import GEKKO as gk
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import sqrt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# create a function for finding the model values
def model_values(dataset, dataset2):
    m = gk()
    x = m.Param(value=dataset)
    a = m.FV(value=0.1)
    b = m.FV(value=0.1)
    c = m.FV(value=0.1)

    a.STATUS = 1
    b.STATUS = 1
    c.STATUS = 1

    y = m.CV(value=dataset2)
    y.FSTATUS = 1
    # m.Equation(y == (a * x) + (b * x**2) + (c * x**3))
    m.Equation(y == a + (b * x) + (c * x ** 2))
    m.options.IMODE = 2
    m.solve(disp=False)

    error1 = numpy.square(dataset2 - y.value).sum() / (len(dataset)) ** 2
    error = math.sqrt(error1)

    return a.value[0], b.value[0], c.value[0], x.value, y.value, error


# load the dataset
cols = [4, 5, 9, 17, 18]

dataframe1 = read_excel('Trans Oil Datasheet.xlsx', sheet_name='#29', usecols=cols)
dataframe2 = read_excel('Trans Oil Datasheet.xlsx', sheet_name='#30', usecols=cols)
dataframe3 = read_excel('Trans Oil Datasheet.xlsx', sheet_name='#31', usecols=cols)
dataframe4 = read_excel('Trans Oil Datasheet.xlsx', sheet_name='#32', usecols=cols)
data29co2m = dataframe1['CO2'].values
data29com = dataframe1['CO'].values
data29nm = dataframe1['N2'].values
data29Toil = dataframe1['T TOP OIL'].values
data29time = dataframe1['Time'].values
data30co2 = dataframe2['CO2'].values
data30com = dataframe2['CO'].values
data30n = dataframe2['N2'].values
data30Toil = dataframe2['T TOP OIL'].values
data30time = dataframe2['Time'].values
data31co2 = dataframe3['CO2'].values
data31co = dataframe3['CO'].values
data31n = dataframe3['N2'].values
data31Toil = dataframe3['T TOP OIL'].values
data31time = dataframe3['Time'].values
data32co2 = dataframe4['CO2'].values
data32co = dataframe4['CO'].values
data32n = dataframe4['N2'].values
data32Toil = dataframe4['T TOP OIL'].values
data32time = dataframe4['Time'].values

data29nm = data29nm.astype('float32')
data29nm = data29nm.reshape(5002, 1)
data29n = data29nm

# train the normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(data29n)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

# train the standardization
data30co = scaler.transform(data29n)
stand = StandardScaler()
stand = stand.fit(data29n)
data29n = stand.transform(data29n)
print('Mean: %f, StandardDeviation: %f' % (stand.mean_, sqrt(stand.var_)))

# split into train and test sets
train_size = int(len(data29n) * 0.75)
test_size = len(data29n) - train_size
train, test = data29n[0:train_size, :], data29n[train_size:len(data29n), :]

# reshape dataset
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(12, input_dim=look_back, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(data29n)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(data29n)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(data29n) - 1, :] = testPredict

'''
# plot baseline and predictions
fig1 = plt.figure()
fig1.set_figheight(15)
fig1.set_figwidth(8)

ax1 = fig1.add_subplot(111, projection='3d')
line1 = ax1.scatter(data29co2m, data29Toil, data29time, c='r', marker='x')
line2 = ax1.scatter(data30co2, data30Toil, data30time, c='g', marker='x')
line3 = ax1.scatter(data31co2, data31Toil, data31time, c='b', marker='x')
line4 = ax1.scatter(data32co2, data32Toil, data32time, c='y', marker='x')
plt.title('CO2 Distribution')
ax1.set_xlabel('Concentration')
ax1.set_ylabel('Oil Temperature')
ax1.set_zlabel('Time(history)')
ax1.legend((line1, line2, line3, line4), ('C02 for #29', 'C02 for #30', 'CO2 for #31', 'CO2 for #32'))
print("")

fig2 = plt.figure()
fig2.set_figheight(15)
fig2.set_figwidth(8)

ax2 = fig2.add_subplot(111, projection='3d')
line5 = ax2.scatter(data29co, data29Toil, data29time, c='r', marker='x')
line6 = ax2.scatter(data30co, data30Toil, data30time, c='g', marker='x')
line7 = ax2.scatter(data31co, data31Toil, data31time, c='b', marker='x')
line8 = ax2.scatter(data32co, data32Toil, data32time, c='y', marker='x')
plt.title('CO Distribution')
ax2.set_xlabel('Concentration')
ax2.set_ylabel('Oil Temperature')
ax2.set_zlabel('Time(history)')
ax2.legend((line5, line6, line7, line8), ('C0 for #29', 'C0 for #30', 'CO for #31', 'CO for #32'))
print("")

fig3 = plt.figure()
fig3.set_figheight(15)
fig3.set_figwidth(8)
ax3 = fig3.add_subplot(111, projection='3d')
line9 = ax3.scatter(data29n, data29Toil, data29time, c='r', marker='x')
line10 = ax3.scatter(data30n, data30Toil, data30time, c='g', marker='x')
line11 = ax3.scatter(data31n, data31Toil, data31time, c='b', marker='x')
line12 = ax3.scatter(data32n, data32Toil, data32time, c='y', marker='x')
plt.title('N2 Distribution')
ax3.set_xlabel('Concentration')
ax3.set_ylabel('Oil Temperature')
ax3.set_zlabel('Time(history)')
ax3.legend((line9, line10, line11, line12), ('N2 for #29', 'N2 for #30', 'N2 for #31', 'N2 for #32'))
print("")

'''
# modelling for data 29
X_path1 = data29Toil[180:1475]
Y_path1 = data29nm[180:1475]
X_path2 = data29Toil[1476:2750]
Y_path2 = data29nm[1476:2750]
X_path3 = data29Toil[2751:3650]
Y_path3 = data29nm[2751:3650]
X_path4 = data29Toil[3651:5001]
Y_path4 = data29nm[3651:5001]
A1, B1, C1, X1, Y1, E1 = model_values(X_path1, Y_path1)
A2, B2, C2, X2, Y2, E2 = model_values(X_path2, Y_path2)
A3, B3, C3, X3, Y3, E3 = model_values(X_path3, Y_path3)
A4, B4, C4, X4, Y4, E4 = model_values(X_path4, Y_path4)

print([A1, B1, C1], '\n', [A2, B2, C2], '\n', [A3, B3, C3], '\n', [A4, B4, C4])
'''
#modelling for data 30
X_pat1 = data30Toil[180:1475]
Y_pat1 = data30com[180:1475]
X_pat2 = data30Toil[1476:2750]
Y_pat2 = data30com[1476:2750]
X_pat3 = data30Toil[2700:3550]
Y_pat3 = data30com[2700:3550]
X_pat4 = data30Toil[3651:5050]
Y_pat4 = data30com[3651:5050]
A1, B1, C1, X1, Y1, E1 = model_values(X_pat1, Y_pat1)
A2, B2, C2, X2, Y2, E2 = model_values(X_pat2, Y_pat2)
A3, B3, C3, X3, Y3, E3 = model_values(X_pat3, Y_pat3)
A4, B4, C4, X4, Y4, E4 = model_values(X_pat4, Y_pat4)
'''
# print([A1, B1, C1], '\n', [A2, B2, C2], '\n', [A3, B3, C3], '\n', [A4, B4, C4])
# print(E1, E2, E3, E4)
print('The coefficients for the first section is %f for a, %f for b and %f for c.' % (A1, B1, C1))
print('The error value  for the first section calculated using the RMSE method is %f.\n\n' % (E1 / 101))
print('The coefficients for the second section is %f for a, %f for b and %f for c.' % (A2, B2, C2))
print('The error value  for the second section calculated using the RMSE method is %f.\n\n' % (E2 / 25))
print('The coefficients for the third section is %f for a, %f for b and %f for c.' % (A3, B3, C3))
print('The error value  for the third section calculated using the RMSE method is %f.\n\n' % (E3 / 81))
print('The coefficients for the fourth section is %f for a, %f for b and %f for c.' % (A4, B4, C4))
print('The error value  for the fourth section calculated using the RMSE method is %f.\n' % (E4 / 31))

fip = plt.figure()
fip.set_figheight(10)
fip.set_figwidth(10)
plt.subplot(111)
plt.title('N2 Distribution for #29')
# lin1 = plt.plot(data29Toil, data30com, 'r-', linewidth=0.3)
lin2 = plt.plot(data29Toil, data29nm, 'g-', linewidth=0.2)
# lin3 = plt.plot(data31Toil, data31co, 'b-', linewidth=0.3)
# lin4 = plt.plot(data32Toil, data32co, 'y-', linewidth=0.3)
plt.plot(X1, Y1)
plt.plot(X2, Y2)
plt.plot(X3, Y3)
plt.plot(X4, Y4)
plt.xlabel('Temperature')
plt.ylabel('Concentration')
# plt.legend([lin1, lin2, lin3, lin4], ['CO for #29', 'CO for #30', 'CO for #31', 'CO for #32'])
plt.grid(axis='both')
print('')

fid = plt.figure()
fid.set_figheight(10)
fid.set_figwidth(10)
plt.subplot(111)
plt.title('N2 Distribution for #29')
# lin1 = plt.plot(data29Toil, data29com, 'r-', linewidth=0.3)
lin2 = plt.plot(data29Toil, data29nm, 'g-', linewidth=0.2)
# lin3 = plt.plot(data31Toil, data31co, 'b-', linewidth=0.3)
# lin4 = plt.plot(data32Toil, data32co, 'y-', linewidth=0.3)
# plt.plot(X1, Y1)
# plt.plot(X2, Y2)
# plt.plot(X3, Y3)
# plt.plot(X4, Y4)
plt.xlabel('Temperature')
plt.ylabel('Concentration')
# plt.legend([lin1, lin2, lin3, lin4], ['CO for #29', 'CO for #30', 'CO for #31', 'CO for #32'])
plt.grid(axis='both')
print('')

fir = plt.figure()
fir.set_figheight(10)
fir.set_figwidth(20)
plt.subplot(121)
plt.plot(X_path1, Y_path1, 'r-', linewidth=0.2)
plt.plot(X1, Y1)
plt.xlabel('Temperature')
plt.ylabel('Concentration')
plt.grid(axis='both')
plt.title('First Quarter Line Plot with model for N2 in Transfer Matter 29')
print('')

plt.subplot(122)
plt.plot(X_path2, Y_path2, 'r-', linewidth=0.2)
plt.plot(X2, Y2)
plt.xlabel('Temperature')
plt.ylabel('Concentration')
plt.grid(axis='both')
plt.title('Second Quarter Line Plot with model for N2 in Transfer Matter 29')
print('')

fie = plt.figure()
fie.set_figheight(10)
fie.set_figwidth(20)
plt.subplot(121)
plt.plot(X_path3, Y_path3, 'r-', linewidth=0.2)
plt.plot(X3, Y3)
plt.xlabel('Temperature')
plt.ylabel('Concentration')
plt.grid(axis='both')
plt.title('Third Quarter Line Plot with model for N2 in Transfer Matter 29')
print('')

plt.subplot(122)
plt.plot(X_path4, Y_path4, 'r-', linewidth=0.2)
plt.plot(X4, Y4)
plt.xlabel('Temperature')
plt.ylabel('Concentration')
plt.grid(axis='both')
plt.title('Fourth Quarter Line Plot with model for N2 in Transfer Matter 29')
print('')

fik = plt.figure()
fik.set_figheight(10)
fik.set_figwidth(20)
plt.subplot(121)
plt.title(dataframe1.columns[2] + ' Concentration for #29')
plt.plot(data29n, 'r', label=dataframe1.columns[2] + " Full Data", linewidth=0.3)
plt.plot(trainPredictPlot, 'y', label=dataframe1.columns[2] + " Train Data", linewidth=0.3)
plt.plot(testPredictPlot, 'g', label=dataframe1.columns[2] + " Test Data", linewidth=0.3)
plt.legend()
plt.grid(axis='both')
plt.xlabel('Data Size')
plt.ylabel('Training Range')
plt.grid(axis='both')
print("")

plt.subplot(122)
plt.title(dataframe1.columns[2] + ' Loss / Mean Squared Error for #29')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.grid(axis='both')
plt.xlabel('MSR')
plt.ylabel('LOSS')
plt.show()
