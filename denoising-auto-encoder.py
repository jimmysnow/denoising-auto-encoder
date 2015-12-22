# ------- TENSORFLOW --------- #
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import sys, os, numpy
from collections import defaultdict
from collections import OrderedDict

# ------------------ General ------------------- #
IMG_DIR = 'VOC2007' + os.sep + 'JPEGImages' + os.sep
LBL_DIR = 'VOC2007' + os.sep + 'ImageSets' + os.sep + 'Main' + os.sep

# ------------------ Training ------------------ #
TRAIN_DIR = 'train' + os.sep + IMG_DIR
TRAIN_LABELS = 'train' + os.sep + LBL_DIR

# ------------------ Testing ------------------- # 
TEST_DIR = 'test' + os.sep + IMG_DIR
TEST_LABELS = 'test' + os.sep + LBL_DIR

# ------------------ Misc ------------------- # 
NUM_CLASSES = 20

print(TRAIN_DIR)
print(TRAIN_LABELS)
print(TEST_DIR)
print(TEST_LABELS)

# Generic read data from our file structure...
def get_data(DIR, LABELS, key):
	imgs = {}
	labels = defaultdict(list)
	for root, dirs, files in os.walk(LABELS, topdown=False):
		for f in files: 
			if '_' + key in f: 
				label = f.split('_')[0] 
				f = os.path.abspath(LABELS + f)
				pos_ex = [l.split()[0] for l in open(f).readlines() if l.split()[-1] == '1']
				for img in pos_ex:
					labels[label].append(img)
					imgs[img] = os.path.abspath(DIR + img + '.jpg')
	return imgs, labels

# Dig through for the training data...
def get_train_data(): return get_data(TRAIN_DIR, TRAIN_LABELS, "trainval.txt")		

# Dig through for the testing data...
def get_test_data(): return get_data(TEST_DIR, TEST_LABELS, "test.txt")		

def get_one_hot_label_list(label_dict):
	keys = OrderedDict(sorted(label_dict.items())).keys()
	one_hot_lookup = {}
	for i in range(len(keys)):
		one_hot = numpy.zeros(NUM_CLASSES)
		one_hot[i] = 1
		one_hot_lookup[keys[i]] = one_hot
	one_hot_list = []
	for label in label_dict.keys():
		vals = label_dict[label]
		for v in vals:
			one_hot_list.append((v, one_hot_lookup[label]))
	return one_hot_list

def read_img_data(one_hot_dict, file_dict):
	labels = []
	files = []
	img_files = []
	reader = tf.WholeFileReader()
	print('Assembling files...')
	for (img, label_vec) in one_hot_dict:
		files.append(file_dict[img])
		labels.append(label_vec)
	labels = numpy.asarray(labels)
	queue = tf.train.string_input_producer(files)
	jpg_key, jpg_img = reader.read(queue)
	jpg_img = tf.image.decode_jpeg(jpg_img)
	init = tf.initialize_all_variables()
	# Run session...
	print('Starting tensorflow session...')
	with tf.Session() as sess:
		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		for i in range(len(files)):
			jpg = jpg_img.eval()
			img_files.append(jpg)
		coord.request_stop()
		coord.join(threads)
		# build numpy arrays
		print('Reconstructing arrays...')
		img_files = numpy.asarray(img_files)
		img_files = tf.convert_to_tensor(img_files)
		tf.Print(img_files)
		img_files = tf.image.resize_images(img_files, 256, 256)
	print('Done!')	
	return img_files, labels

# ------------------------------------- #
# --------- TRAIN & TEST SETS --------- #
# ------------------------------------- #

train_imgs, train_labels = get_train_data()
test_imgs, test_labels = get_test_data()

train_labels_one_hot = get_one_hot_label_list(train_labels)
test_labels_one_hot = get_one_hot_label_list(test_labels)

# ------------------------------------- #
# --------- PROGRAM START ------------- #
# ------------------------------------- #

read_img_data(train_labels_one_hot, train_imgs)


