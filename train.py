import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

def main():
	# read train.csv file
	raw = pd.read_csv("train.csv")
	raw.drop(raw.columns[[0, 1]], axis=1, inplace=True)
	raw = raw.as_matrix()
	raw = np.ravel(raw)

	# define x , y
	x = []
	y = []

	month_in_data=11
	####################################### START TODO's ######################################
	for m in range(month_in_data):
		hour_prior = 5 #TODO #how many hours to use to predict next PM2.5
		for d in range(480-1-hour_prior):
			# x is an array of pm2.5 of [hour1 hour2 hour3 hour4 hour5 bias]
			tmp = raw[m*480 + d : m*480 + d + hour_prior]
			tmp = np.append(tmp, [1])
			x.append(tmp)
			# y is value of pm2.5 of [hour6]
			y.append(raw[m*480 + d + hour_prior])
	####################################### END TODO's #######################################

	x = np.array(x)
	y = np.array(y)

	# train pm2.5 using linear regression
	####################################### START TODO's #######################################
	# define learning rate
	# small learning rate trains slower but steadily
	l_rate = 0.000000001 #TODO #Try to adjust the learning rate

	# define repeat time
	# large repeat time may get closer to the minimun of loss
	repeat = 100 #TODO

	# here you need to update (w1, w2, w3, w4, w5, b) according to their gradient
	# step 1. init parameters(w1, w2, w3, w4, w5, b) 
	parameters = np.zeros(hour_prior+1)
	for i in range(repeat):
		# step 2. calculate loss
		y_pred = np.dot(x,np.transpose(parameters))
		loss = y_pred - y
		gradient = np.dot(np.transpose(x),loss)
		parameters = parameters - l_rate * gradient
		
		# print cost every iteration
		cost = np.sum(loss**2) / len(x)
		cost = np.sqrt(cost)
		print('iteration: %d | cost: %f' %(i, cost))
		print_parameter(parameters)
	####################################### END TODO's #######################################

	# let's see what you have trained
	print('after training for %d times, you get' %(repeat))
	print_parameter(parameters)

	# un-comment this part of code after you trained
	# you can see how close your predition and correct answer is
	'''
	predicted = np.dot(x,np.transpose([w1,w2,w3,w4,w5,b]))
	fig, ax = plt.subplots()
	ax.scatter(y, predicted)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()
	'''

def print_parameter(parameter):
	for idx in range(len(parameter)-1):
		print('w%d:%.4f ' % (idx+1, parameter[idx]),)
	print('b:%.4f' % (parameter[-1]))

if __name__ == '__main__':
	main()