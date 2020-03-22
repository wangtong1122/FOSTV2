import numpy as np
import os
import glob
from abc import abstractmethod

class DataLoader(object):
	def get_images(self, data_dir):
		print("输入数据路径："+data_dir)
		files = []
		# for ext in ['jpg', 'png', 'jpeg', 'JPG']:
		# print(glob.glob(os.path.join(data_dir, 'img*.jpg')))
		# glob.glob(r"E:\Learn\AI\DL\DataSets\ch4_training_images\*.jpg")
		files.extend(glob.glob(r"E:\Learn\AI\DL\DataSets\ch4_training_images\*.jpg"))
		return files

	@abstractmethod
	def load_annotation(self, gt_file):
		print("reimplement by particular data loader")
		pass
