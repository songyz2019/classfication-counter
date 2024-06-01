from torch import flatten, argmax, Tensor
from itertools import product
import numpy as np
from matplotlib import pyplot as pl
import holoviews as hv
import pandas as pd

class ClassificationCounter:
    """计数类,用来统计分类的准确率(rate)"""

    def __init__(self):
        self.data = {}  # {类名: {类名: 数量,},}

    def count(self, truth, predict):
        """计数"""
        if truth not in self.data:
            self.data[truth] = {}
        if predict not in self.data[truth].keys():
            self.data[truth][predict] = 0

        self.data[truth][predict] += 1

    def total(self, truth):
        return sum(self.data[truth].values())

    def correct(self, truth) -> int:
        """正确样本数"""
        if not truth in self.data[truth]:
            return 0
        return self.data[truth][truth]

    def rate(self) -> dict:
        """分类正确率
        @return {'A': 0.5, 'B': 0.3}
        """
        return {truth: self.correct(truth) / self.total(truth) for truth in self.data.keys()}

    def frac(self) -> dict:
        """分类正确率的分数形式
        @return {'A': "12/30", 'B': "16/31"}
        """
        return {truth: f"{self.correct(truth)}/{self.total(truth)}" for truth in self.data.keys()}

    def average_rate(self):
        """平均准确率"""
        correct_rate = self.rate().values()
        return sum(correct_rate) / len(correct_rate)

    def overall_total(self) -> int:
        """总样本数"""
        return sum([self.total(v) for v in self.data.keys()])

    def overall_correct(self) -> int:
        """总正确样本数"""
        return sum([self.correct(v) for v in self.data.keys()])

    def overall_rate(self) -> float:
        """总正确率"""
        return self.overall_correct() / self.overall_total()

    def overall_frac(self):
        """总正确率的分数形式"""
        return f"{self.overall_correct()}/{self.overall_total()}"

    def confusion_matrix(self):
        N = len(self.data)
        matrix = np.zeros((N, N), dtype='int')
        keys = sorted(self.data.keys())
        for i, j in product(range(N), range(N)):
            try:
                matrix[i][j] = self.data[keys[i]][keys[j]]
            except KeyError:
                matrix[i][j] = 0
        return matrix

    def kappa(self):
        m = self.confusion_matrix()
        p0 = m.trace() / m.sum()
        pe = np.dot(m.sum(0), m.sum(1)) / (m.sum() ** 2)
        return (p0 - pe) / (1 - pe)

    def __call__(self, truth, predict, de_onehot=True):
        """进行计数

        如果y是张量,会自动展平进行反OneHot编码
        @param truth 样本真值
        @param predict 样本预测值
        @param de_onehot 自动展平进行反OneHot编码(y为torch.Tensor时)
        @return 日志信息
        """
        if de_onehot and isinstance(truth, Tensor) and truth.size().numel() > 1:
            truth = int(argmax(flatten(truth)))
        if de_onehot and isinstance(predict, Tensor) and predict.size().numel() > 1:
            predict = int(argmax(flatten(predict)))

        self.count(truth, predict)
        return f"{self.overall_frac()} = {self.overall_rate():.2f}"

    def plot_classified(self, key_map=lambda x: x, figure=True):
        if figure:
            pl.figure()

        keys = list(self.data.keys())
        x = [key_map(k) for k in keys]
        pl.bar(x, [1 for k in keys], label="incorrect")
        pl.bar(x, list(self.rate().values()), label="correct")
        pl.legend()
        return pl.gcf()


    def plot_classified_frac(self, key_map=lambda x: x, figure=True):
        if figure:
            pl.figure()

        keys = list(self.data.keys())
        x = [key_map(k) for k in keys]
        pl.bar(x, [self.total(k) for k in keys], label="incorrect")
        pl.bar(x, [self.correct(k) for k in keys], label="correct")
        pl.legend()
        return pl.gcf()


    def plot_overview(self, figure=True):
        if figure:
            pl.figure()

        d = {
            "Overall Accuracy (OA)": self.overall_rate(),
            "Average Accuracy (AA)": self.average_rate(),
            "Kappa": self.kappa(),
        }
        pl.bar(d.keys(), d.values())
        pl.legend()
        return pl.gcf()

    def plot_confusion(self):
        confusion = self.confusion_matrix()
        N = confusion.shape[0]

        origin_extension = hv.Store.current_backend
        hv.extension('matplotlib')

        img = hv.HeatMap(
            [(i,j,confusion[i][j]) for i,j in product(range(N), range(N))]
        ).sort()

        hv.extension(origin_extension)
        return hv.render(img.opts(fig_inches=N))

    def to_dataframe(self):
        keys = sorted(self.data.keys())
        return pd.DataFrame(self.confusion_matrix(), columns=keys)

    def report(self):
        return f"""
```text
---------------Counter-------------------
data: {self.data}
confusion: {self.confusion_matrix()}
classified(CA):{self.frac()} = {self.rate()}
overall(OA):   {self.overall_frac()} =  {self.overall_rate()}
average(AA):   {self.average_rate()}
kappa: {self.kappa()}
-----------------------------------------
```
    """.strip()

    def __str__(self):
        return self.report()

    @classmethod
    def example(cls):
        counter = ClassificationCounter()
        counter.data = {
            "A": {"A": 15, "B": 7, "C": 0, "D": 8},
            "B": {"A": 0, "B": 17, "C": 11, "D": 0},
            "C": {"A": 0, "B": 17, "C": 11, "D": 0},
            "D": {"A": 0, "B": 4, "C": 19, "D": 0}
        }
        return counter


# ------------------
#    def plot_sankey(self, figure=True):
#        """
#        绘制Sankey图,描述分类流动情况
#
#        这是一个Plotly绘图,可能在非Jupyter环境下无法显示
#        """
#        keys = list(self.data.keys())
#        N = len(keys)
#        source = []
#        target = []
#        value  = []
#        for i, j in product(keys, keys):
#            source.append(keys.index(i))
#            target.append(keys.index(j) + N)
#            value.append(self.data[i][j])
#
#        s = go.Sankey(
#            node = {
#                "label": [*keys, *keys],
#                "color": ['green']*N + ['blue']*N
#            },
#            link= {
#                "source": source,
#                "target": target,
#                "value" : value,
#            }
#        )
#
#        if figure:
#            fig = go.Figure(s)
#            return fig
#        else:
#            return s
