import sys
import os
import numpy

class DataSet():
	def __init__(self, data_dir, input_dim, name):
		self.train_ptr = 0
		self.test_ptr = 0
		self.input_dim = input_dim
		(lbl, dat) = self.read_data_sets(data_dir, input_dim, name)
		self.train_labels = lbl
		self.train_data = dat
		self.num_train_examples = len(self.train_data)
		(lbl, dat) = self.read_data_sets(data_dir, input_dim, name, type="test")
		self.test_labels = lbl
		self.test_data = dat
		self.num_test_examples = len(self.test_data)
		
	def read_data_sets(self, data_dir, input_dim, name, type="train"):
		lbl = []
		dat = []
		for root, dirs, files in os.walk(data_dir, topdown=False):
			for f in files: 
				f = os.path.abspath(root + os.sep + f)
				should_be = os.path.abspath(root + os.sep + name + '-' + type + '.csv')
				if f == should_be:
					print(f)
					items = [l.strip().split() for l in open(f).readlines()]
					for item in items:
						output_dim = len(item) - input_dim
						y = numpy.array(item[0:output_dim], dtype='|S4')
						y = y.astype(numpy.int32)[0]
						x = numpy.array(item[output_dim:], dtype='|S4')
						x = x.astype(numpy.float32)
						lbl.append(y)
						dat.append(x)
		return (lbl, dat)
	
	def next_batch(self, batch_size, type="train"):
		batch = []
		labels = []
		if type is "train": 
			lim = self.train_ptr + batch_size
			lim = min(lim, self.num_train_examples)
			batch = self.train_data[self.train_ptr:lim]
			labels = self.train_labels[self.train_ptr:lim]
			self.train_ptr += lim
			if self.train_ptr >= self.num_train_examples: 
				self.train_ptr = 0
		elif type is "test": 
			lim = self.test_ptr + batch_size
			lim = min(lim, self.num_test_examples)
			batch = self.test_data[self.test_ptr:lim]
			labels = self.train_labels[self.test_ptr:lim]
			self.test_ptr += lim
			if self.test_ptr >= self.num_test_examples: 
				self.test_ptr = 0
		return batch, labels




