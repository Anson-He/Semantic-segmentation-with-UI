import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QListWidgetItem, QPushButton
from flags import *
from scripts.demo import demo

class MyItem(QListWidgetItem):
    def __init__(self, name=None, parent=None):
        super(MyItem, self).__init__(name, parent=parent)
        self.setIcon(QIcon('icons/color.png'))
        self.setSizeHint(QSize(60, 60))  # size

    def get_params(self):
        protected = [v for v in dir(self) if v.startswith('_') and not v.startswith('__')]
        param = {}
        for v in protected:
            param[v.replace('_', '', 1)] = self.__getattribute__(v)
        return param

    def update_params(self, param):
        for k, v in param.items():
            if '_' + k in dir(self):
                self.__setattr__('_' + k, v)


class SemanticItem(MyItem):
    def __init__(self, parent=None):
        super().__init__('语义', parent=parent)
        self._data = 0
        self._model = 0
    def __call__(self, img):
        # if self._kind == MEAN_FILTER:
        #     img = cv2.blur(img, (self._ksize, self._ksize))
        # elif self._kind == GAUSSIAN_FILTER:
        #     img = cv2.GaussianBlur(img, (self._ksize, self._ksize), self._sigmax)
        # elif self._kind == MEDIAN_FILTER:
        #     img = cv2.medianBlur(img, self._ksize)
        model = ['fcn8s_vgg16_', 'fcn16s_vgg16_', 'fcn32s_vgg16_', 'fcn_resnet50_', 'psp_resnet50_', 'deeplabv3_resnet50_']
        data = ['voc', 'citys']
        model_name = model[self._model]+data[self._data]
        f = open('./url.txt', 'r')
        a = f.readline()
        f.close()
        print(a)
        if data[self._data] == 'voc':
            dataset = 'pascal_voc'
        else:
            dataset = 'citys'
        model_name = model[self._model] + data[self._data]
        mask = demo(model_name, dataset, a, outdir='./scripts/eval', save_folder='./models', local_rank=0)
        return mask


