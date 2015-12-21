import sys
import os

class DataSet():
    def __init__(self, data_dir, input_dim, name):
        self.pointer = 0
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
                        lbl.append(item[0:output_dim])
                        dat.append(item[output_dim:])
        return (lbl, dat)
    
    def next_batch(self, batch_size, type="train"):
        batch = []
        labels = []
        lim = self.pointer + batch_size
        if type is "train": 
            lim = min(lim, self.num_train_examples)
            batch = self.train_data[self.pointer:lim]
            labels = self.train_labels[self.pointer:lim]
            self.pointer += lim
            if self.pointer >= self.num_train_examples: 
                self.pointer = 0
        elif type is "test": 
            lim = min(lim, self.num_test_examples)
            batch = self.test_data[self.pointer:lim]
            labels = self.train_labels[self.pointer:lim]
            self.pointer += lim
            if self.pointer >= self.num_test_examples: 
                self.pointer = 0
        return batch, labels