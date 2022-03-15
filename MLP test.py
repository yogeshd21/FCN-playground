import pandas as pd
#from keras.datasets import cifar100 #Replace use
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop

def load_dataset():
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	#(trainX, trainY), (testX, testY) = cifar100.load_data(label_mode="fine") #Replace use
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

def prep_pixels(train, test):
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm

def define_model2(hnodes, dr, optm):
	model = Sequential()
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(100, activation='softmax'))
	if optm == 'SGD':
		opt = SGD(lr=0.001, momentum=0.9)
	elif optm == 'ADAM':
		opt = Adam(learning_rate=0.001)
	elif optm == 'RMSProp':
		opt = RMSprop(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def define_model3(hnodes, dr, optm):
	model = Sequential()
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(100, activation='softmax'))
	if optm == 'SGD':
		opt = SGD(lr=0.001, momentum=0.9)
	elif optm == 'ADAM':
		opt = Adam(learning_rate=0.001)
	elif optm == 'RMSProp':
		opt = RMSprop(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def define_model4(hnodes, dr, optm):
	model = Sequential()
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(100, activation='softmax'))
	if optm == 'SGD':
		opt = SGD(lr=0.001, momentum=0.9)
	elif optm == 'ADAM':
		opt = Adam(learning_rate=0.001)
	elif optm == 'RMSProp':
		opt = RMSprop(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def define_model5(hnodes, dr, optm):
	model = Sequential()
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(hnodes, activation='relu'))
	model.add(Dropout(dr))
	model.add(Dense(100, activation='softmax'))
	if optm == 'SGD':
		opt = SGD(lr=0.001, momentum=0.9)
	elif optm == 'ADAM':
		opt = Adam(learning_rate=0.001)
	elif optm == 'RMSProp':
		opt = RMSprop(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def run_main():
	trainX, trainY, testX, testY = load_dataset()
	trainX, testX = prep_pixels(trainX, testX)

	df = pd.read_csv('./Data.csv')
	for i in range(len(df)):
		if df['No. of Layers'][i] == 2:
			model = define_model2(df['No. of Hidden Nodes'][i], df['Dropout Rate'][i], df['Optimizer'][i])
		elif df['No. of Layers'][i] == 3:
			model = define_model3(df['No. of Hidden Nodes'][i], df['Dropout Rate'][i], df['Optimizer'][i])
		elif df['No. of Layers'][i] == 4:
			model = define_model4(df['No. of Hidden Nodes'][i], df['Dropout Rate'][i], df['Optimizer'][i])
		elif df['No. of Layers'][i] == 5:
			model = define_model5(df['No. of Hidden Nodes'][i], df['Dropout Rate'][i], df['Optimizer'][i])

		model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=0)

		_, acc = model.evaluate(testX, testY, verbose=0)
		df.loc[i, 'Accuracy'] = round(acc * 100.0, 3)
		print(i, '> %.3f' % (acc * 100.0))
	df.to_csv('Ans_CIFAR10.csv', index=False)
	#df.to_csv('Ans_CIFAR100.csv', index=False) #Replace use for CIFAR-100 Dataset

run_main()