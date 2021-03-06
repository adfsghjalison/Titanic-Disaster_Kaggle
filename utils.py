from flags import FLAGS
import numpy as np
import csv
import os

class utils():
    def __init__(self, args):
        self.batch_size = FLAGS.batch_size
        self.data_dir = FLAGS.data_dir
        self.train_data = os.path.join(self.data_dir, 'train')
        self.test_data = os.path.join(self.data_dir, 'test')
        self.val_data = os.path.join(self.data_dir, 'val')
        self.xv_size = self.get_xv_size()

    def get_xv_size(self):
        cf = csv.reader(open(self.train_data))
        v = next(cf, None)
        return len(v) - 2

    def get_train_batch(self):
        x = []
        y = []
        cnt = 0
        while(True):
            cf = csv.reader(open(self.train_data))
            for v in cf:
                x.append(v[2:])
                y.append([v[1]])
                cnt += 1
                if cnt >= self.batch_size:
                    x = np.array(x)
                    y = np.array(y)
                    yield x, y
                    x = []
                    y = []
                    cnt = 0

    def get_test_batch(self, mode='test'):
        x = []
        y = []
        id = []
        cnt = 0

        if mode == 'test':
          cf = csv.reader(open(self.test_data))
        else:
          cf = csv.reader(open(self.val_data))
          
        for v in cf:
            x.append(v[2:])
            y.append([v[1]])
            id.append(int(float(v[0])))
            cnt += 1
            if cnt >= self.batch_size:
                x = np.array(x)
                y = np.array(y)
                yield x, y, id
                x = []
                y = []
                id = []
                cnt = 0
        if cnt > 0:
            x = np.array(x)
            y = np.array(y)
            yield x, y, id

def write_test(id, pre):
    cf = csv.writer(open(os.path.join(FLAGS.data_dir, 'prediction.csv'), 'w'))
    cf.writerow(['PassengerID', 'Survived'])
    for i, p in zip(id, pre):
      cf.writerow([i, p])

