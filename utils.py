import numpy as np

counter_int = 60


class DCache:
    # TODO: AUGMENT IT SUCH THAT IT WORKS FOR MULTIPLE

    def __init__(self, size=20, thres=3, buffer=False):
        """
        :param size: int, size of the dampening cache
        :param thres: float, threshold for valid data caching, ignore signal if |x - miu_x| > thres * std
        :param buffer: boolean, for whether keeping a dynamic buffer
        """
        self.size = size
        self.thres = thres
        self.counter = 0

        if buffer:
            self.cache = []
        else:
            self.avg = 0
            self.std = 0

    def __len__(self):
        return self.size

    def add(self, signal):
        if signal != 0:
            if self.counter < self.size:
                print(self.avg, self.avg * (self.counter - 1), (self.avg * self.counter + signal) / (self.counter + 1))
                self.avg = (self.avg * self.counter + signal) / (self.counter + 1)
                diff2 = (signal - self.avg) ** 2
                self.std = (diff2 + self.std * self.counter) / (self.counter+1)

            elif signal - self.avg < np.sqrt(self.std * self.thres):
                print(self.avg, self.avg * (self.size - 1), (self.avg * (self.size - 1) + signal) / self.size)
                self.avg = (self.avg * (self.size - 1) + signal) / self.size
                diff2 = (signal - self.avg) ** 2
                self.std = (diff2 + self.std * (self.size - 1)) / self.size
            self.counter += 1

    def get_val(self):
        return self.avg


def second_to_hmt(sec):
    h, sec = sec // 3600, sec % 3600
    m, sec = sec // 60, sec % 60
    return h, m, sec



