import os
import numpy as np
from .data_loader import DataLoader
from .data_utils import label_to_array
from __future__ import print_function
from __future__ import unicode_literals

import json
import pprint
import settings

from pythonapi import anno_tools


class CTW_loader(DataLoader):

    def __init__(self, shuffle=False):
        super(CTW_loader, self).__init__()
        self.shuffle = shuffle  # shuffle the polygon
        print('读取所有训练集数据标签')
        self.dict = {}
        with open(settings.TRAIN) as f:
            for line in f.readlines():
                anno = json.loads(line)
                imageid = anno['image_id']
                self.dict[imageid] = anno

    # 根据图片文件名称加载对应的标签文件
    def load_annotation(self, im_id):
        text_polys = []
        text_tags = []
        labels = []
        ano = self.dict[im_id]

        for line in f.readlines():
                try:
                    if len(line) > 9:
                        label = line[8]
                        for i in range(len(line) - 9):
                            label = label + "," + line[i + 9]
                    else:
                        label = line[-1]

                    temp_line = list(map(eval, line[:8]))
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, temp_line)

                    text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    if label == '*' or label == '###' or label == '':
                        text_tags.append(True)
                        labels.append([-1])
                    else:
                        labels.append(label_to_array(label))
                        text_tags.append(False)
                except Exception as e:
                    print(e)
                    continue
        text_polys = np.array(text_polys)
        text_tags = np.array(text_tags)
        labels = np.array(labels)

        index = np.arange(0, text_polys.shape[0])
        if self.shuffle:
            np.random.shuffle(index)
            text_polys = text_polys[index]
            text_tags = text_tags[index]
            labels = labels[index]

        if text_polys.shape[0] > 32:
            text_polys = text_polys[:32]
            text_tags = text_tags[:32]
            labels = labels[:32]
        labels = labels.tolist()
        return text_polys, text_tags, labels
