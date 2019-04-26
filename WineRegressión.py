import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.set_printoptions(precision=4, suppress=True)

dataWineRed = pd.read_csv("WineQualityRed.csv", sep=",").iloc[:, 1:]
dataWineWhite = pd.read_csv("WineQualityWhite.csv", sep=",").iloc[:, 1:]

for i in dataWineRed.keys():
    print("\t", "{:.4f}".format(dataWineRed[i].min()),
          "\t", "{:.4f}".format(dataWineRed[i].mean()),
          "\t", "{:.4f}".format(dataWineRed[i].max()), "\t", i)

print()

for i in dataWineWhite.keys():
    print("\t", "{:.4f}".format(dataWineWhite[i].min()),
          "\t", "{:.4f}".format(dataWineWhite[i].mean()),
          "\t", "{:.4f}".format(dataWineWhite[i].max()), "\t", i)

print()

print(dataWineRed.head())
print()
print(dataWineWhite.head())
print()

shuffleWineRed = dataWineRed.sample(frac=1)
shuffleWineWhite = dataWineWhite.sample(frac=1)

numTrainRed = np.floor(0.8*len(shuffleWineRed.index)).astype(np.int)
numTrainWhite = np.floor(0.8*len(shuffleWineWhite.index)).astype(np.int)

trainWineRedX = shuffleWineRed.iloc[:numTrainRed, :-1]
trainWineRedY = shuffleWineRed.iloc[:numTrainRed, -1]
testWineRedX = shuffleWineRed.iloc[numTrainRed:, :-1]
testWineRedY = shuffleWineRed.iloc[numTrainRed:, -1]
trainWineWhiteX = shuffleWineWhite.iloc[:numTrainWhite, :-1]
trainWineWhiteY = shuffleWineWhite.iloc[:numTrainWhite, -1]
testWineWhiteX = shuffleWineWhite.iloc[numTrainWhite:, :-1]
testWineWhiteY = shuffleWineWhite.iloc[numTrainWhite:, -1]

print(trainWineRedX.shape)
print(trainWineRedY.shape)
print(testWineRedX.shape)
print(testWineRedY.shape)
print()

print(trainWineWhiteX.shape)
print(trainWineWhiteY.shape)
print(testWineWhiteX.shape)
print(testWineWhiteY.shape)
print()

matXRed = np.append(np.ones((numTrainRed, 1)), trainWineRedX.values, axis=1)
XporXRed = np.matmul(np.transpose(matXRed), matXRed)
invXporXRed = np.linalg.inv(XporXRed)
XporYRed = np.matmul(np.transpose(matXRed), trainWineRedY.values)
coefBetaRed = np.matmul(invXporXRed, XporYRed)
print(coefBetaRed)
print()

linRegRed = LinearRegression()
linRegRed.fit(trainWineRedX, trainWineRedY)
predictWineRed = linRegRed.predict(testWineRedX)
MSERed = mean_squared_error(testWineRedY, predictWineRed)
print("Linear regression results for the RED Wine")
print("Regression intercept:\t", linRegRed.intercept_)
print("Linear regression coefficients: ", linRegRed.coef_)
print("Mean square error:\t\t", "{:.4F}".format(MSERed))
print("Root mean square error:\t", "{:.4f}".format(np.sqrt(MSERed)))
print()

matXWhite = np.append(np.ones((numTrainWhite, 1)), trainWineWhiteX.values, axis=1)
XporXWhite = np.matmul(np.transpose(matXWhite), matXWhite)
invXporXWhite = np.linalg.inv(XporXWhite)
XporYWhite = np.matmul(np.transpose(matXWhite), trainWineWhiteY.values)
coefBetaWhite = np.matmul(invXporXWhite, XporYWhite)
print(coefBetaWhite)
print()

linRegWhite = LinearRegression()
linRegWhite.fit(trainWineWhiteX, trainWineWhiteY)
predictWineWhite = linRegWhite.predict(testWineWhiteX)
MSEWhite = mean_squared_error(testWineWhiteY, predictWineWhite)
print("Linear regression results for the WHITE Wine")
print("Regression intercept:\t", linRegWhite.intercept_)
print("Linear regression coefficients: ", linRegWhite.coef_)
print("Mean square error:\t\t", "{:.4F}".format(MSEWhite))
print("Root mean square error:\t", "{:.4f}".format(np.sqrt(MSEWhite)))
print()
