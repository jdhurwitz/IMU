import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

class stepcounter:
	def __init__(self, src, cols):
		"""
		src is the path to the raw data file
		cols contains the set of columns we want to pull into the dataframe
		"""
		self.src = src
		self.cols = cols


		#note: need index_col=False to avoid the weird right col shift
		self.IMU_data = pd.read_csv(self.src, header=0, index_col=False, usecols=cols)

		#clip times by removing first 5 digits
		for i in range(len(self.IMU_data['WatchtImeIntervalSince1970'])):
			self.IMU_data['WatchtImeIntervalSince1970'][i] = float(str(self.IMU_data['WatchtImeIntervalSince1970'][i])[5:])

		self.timestamps = self.IMU_data['WatchtImeIntervalSince1970']

		self.filtered_data = self.IMU_data.copy(deep=True)
		for col in cols[1:len(cols)]:
#			print(self.filtered_data[col])
			self.filtered_data[col] = self.applyFilter(10, self.filtered_data[col])




	def printDf(self, nrows=5):
		print(self.IMU_data.head(n=nrows))


	def count(self, arr, ):
		"""
		count the number of "steps". Step is defined as num before is -1
		arr: a 1D np array
		"""
		count = 0
		for i in range(1, len(arr)):
			if (arr[i] == 1 and arr[i-1] == -1):
				count += 1
		return count


	def visualizeAccel(self, offset=300, nsamples=600):
		"""
		Plots the raw (unfiltered) accelerometer only data 
		"""
		x = np.array(self.IMU_data['WatchtImeIntervalSince1970'][offset:nsamples])
#		print(x)
		fig, ax = plt.subplots(3)
		fig.subplots_adjust(hspace = 1)
		for i in range(0,3):
			y = np.array(self.IMU_data[self.cols[i+4]][offset:nsamples])
			ax[i].set_title(self.cols[i+4])
			ax[i].plot(x,y)
		plt.show()


	def applyFilter(self, cutoff, waveform):
		"""
		cutoff value in Hz
		apply a linear IIR filter forward and backward
		"""
		fs = 60 #sampling freq

		w = cutoff/(fs / 2) #normaliztaion 

		#IIR filter coefficients - b=numerator, a=denom
		b, a = signal.butter(5, w, btype='low', analog=False)
		filtered = signal.filtfilt(b, a, waveform)

		return filtered


	def plotFiltered(self, cutoff=10, offset=300, nsamples=600):
		x = np.array(self.IMU_data['WatchtImeIntervalSince1970'][offset:nsamples])
		fig, ax = plt.subplots(3)
		fig.subplots_adjust(hspace = 1)
		for i in range(0,3):
			y = self.applyFilter(cutoff, 
				waveform=np.array(self.IMU_data[self.cols[i+4]][offset:nsamples]))
			ax[i].set_title("Filtered "+self.cols[i+4])
			ax[i].plot(x,y)
		plt.show()


	def applyThresholding(self, y, lag, threshold, influence):
	    signals = np.zeros(len(y))
	    filteredY = np.array(y)
	    avgFilter = [0]*len(y)
	    stdFilter = [0]*len(y)
	    avgFilter[lag - 1] = np.mean(y[0:lag])
	    stdFilter[lag - 1] = np.std(y[0:lag])
	    for i in range(lag, len(y)):
	        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
	            if y[i] > avgFilter[i-1]:
	                signals[i] = 1
	            else:
	                signals[i] = -1

	            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
	            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
	            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
	        else:
	            signals[i] = 0
	            filteredY[i] = y[i]
	            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
	            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

	    return dict(signals = np.asarray(signals),
	                avgFilter = np.asarray(avgFilter),
	                stdFilter = np.asarray(stdFilter))






if __name__ == '__main__':
	path = 'data/Motion-sessions_2019-09-14_15-06-01.csv'
	cols = ['WatchtImeIntervalSince1970', 'WatchGyroX',	'WatchGyroY', 'WatchGyroZ', 'WatchAccX', 'WatchAccY', 'WatchAccZ']
	sc = stepcounter(src=path, cols=cols)
	sc.printDf()


	lag = 30
	threshold = 0.6
	influence = 0
	
	threshold_outputs = sc.applyThresholding(
		sc.filtered_data['WatchGyroY'], 
		lag=lag,
		threshold=threshold,
		influence=influence)
	print(threshold_outputs['signals'][300:350])
	print("num peaks: ", sc.count(threshold_outputs['signals']) )

#	plt.plot(sc.timestamps[300:350], threshold_outputs['signals'][300:350])
#	plt.show()
#	sc.visualizeAccel()	
#	print(sc.IMU_data)
#	sc.plotFiltered()