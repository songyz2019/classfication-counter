import numpy as np
import skimage
import torch
from scipy.sparse import lil_array
from torch import argmax, flatten
import matplotlib.pyplot as pl

class ClassificationMapper:
    def __init__(self, truth):
        """
        将预测结果记录到地图上

        @param truth 稀疏矩阵, 真实值
        """
        self.TRUTH = truth
        self.predict = lil_array(truth.shape, dtype='uint')

    def count(self, location, class_id):
        self.predict[*location] = class_id

    def predict_image(self):
        return skimage.color.label2rgb(self.predict.todense())

    def error_image(self):
        return self.predict.todense() != self.TRUTH.todense()

    def correct_image(self):
        return np.logical_and(self.predict.todense() == self.TRUTH.todense(), self.TRUTH.todense() != 0)

    def result_image(self, underlying=None):
        return skimage.color.label2rgb(self.correct_image() + 2*self.error_image(), colors=['green', 'red'], alpha=0.5, bg_label=0, image=underlying)


    def add_polygen(self):
        pass

    def __call__(self, location, class_id):
        """
        记录坐标和分类

        @param class_id 从1开始的分类id, 如果是数组,自动进行OneHot
        """

        if isinstance(class_id, torch.Tensor) and class_id.size().numel() > 1:
            class_id = int(argmax(flatten(class_id))) + 1

        x,y = location
        if isinstance(x, torch.Tensor):
            x = int(x.item())
            y = int(y.item())

        self.count((x,y), class_id)
