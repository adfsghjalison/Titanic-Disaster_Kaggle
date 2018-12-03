import tensorflow as tf
from utils import utils
import os, csv

class DNN():
    def __init__(self, args):
        self.mode = args.mode
        self.model_dir = args.model_dir
        self.model_path = os.path.join(self.model_dir, 'model')
        self.data_dir = args.data_dir
        self.units = args.units
        self.load = args.load
        self.print_step = args.print_step
        self.save_step = args.save_step
        self.max_step = args.max_step

        self.utils = utils(args)
        self.xv_size = self.utils.xv_size
        self.dp = 0.5

        self.sess = tf.Session()
        self.build()
        self.saver = tf.train.Saver(max_to_keep = 10)

    def build(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.xv_size])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])

        # input layer
        w = tf.Variable(tf.truncated_normal([self.xv_size, self.units[0]]))
        b = tf.Variable(tf.zeros([self.units[0]]))
        a = tf.nn.sigmoid(tf.matmul(self.x, w) + b)
        if self.mode == 'train':
          a = tf.nn.dropout(a, self.dp)
        else:
          a = tf.scalar_mul(self.dp, a)

        # hidden layer
        for i in range(len(self.units)-1):
            w = tf.Variable(tf.truncated_normal([self.units[i], self.units[i+1]]))
            b = tf.Variable(tf.zeros([self.units[i+1]]))
            a = tf.nn.sigmoid(tf.matmul(a, w) + b)
            if self.mode == 'train':
              a = tf.nn.dropout(a, self.dp)
            else:
              a = tf.scalar_mul(self.dp, a)

        # output layer
        w = tf.Variable(tf.truncated_normal([self.units[-1], 1]))
        b = tf.Variable(tf.zeros([1]))
        self.logits = tf.matmul(a, w) + b

        # train
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        #self.op = tf.train.MomentumOptimizer(0.001, 0.8).minimize(self.loss)

        # test
        self.pred = tf.cast(tf.greater(tf.nn.sigmoid(self.logits), 0.65), tf.float32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y), tf.float32))

    def train(self):

        self.sess.run(tf.global_variables_initializer()) 
        step = 1
        acc = 0.0

        if not os.path.exists(self.model_dir):
            os.system("mkdir -p {}".format(self.model_dir))
        
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))
            step = int(ckpt.model_checkpoint_path.split('-')[-1])

        for x, y in self.utils.get_train_batch():
            feed_dict = {self.x : x, self.y : y}
            output = [self.op, self.acc]
            _, acc_temp = self.sess.run(output, feed_dict)

            acc += acc_temp

            if step % self.print_step == 0:
                acc /= self.print_step
                print("Step : {}    Acc : {}".format(step, acc))
                acc = 0.0

            if step % self.save_step == 0:
                print("Saving model ...")
                self.saver.save(self.sess, self.model_path, global_step=step)

            if step >= self.max_step:
                break

            step += 1

    def test(self):
        if self.load != '':
            self.saver.restore(self.sess, self.load)
            print("load model from {} ...".format(self.load))
        else:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))

        cf = csv.writer(open(os.path.join(self.data_dir, 'prediction.csv'), 'w'))
        cf.writerow(['PassengerID', 'Survived'])

        for x, _, id in self.utils.get_test_batch():
            feed_dict = {self.x : x}
            pred = self.sess.run([self.pred], feed_dict)
            for i, p in zip(id, pred[0]):
                cf.writerow([i, int(p[0])])

    def val(self):
        if self.load != '':
            self.saver.restore(self.sess, self.load)
            print("load model from {} ...".format(self.load))
        else:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("load model from {} ...".format(ckpt.model_checkpoint_path))

        acc = 0.0
        cnt = 0

        for x, y, _ in self.utils.get_test_batch(mode='val'):
            feed_dict = {self.x : x, self.y : y}
            acc_temp = self.sess.run([self.acc], feed_dict)
            cnt += len(x)
            acc += acc_temp[0] * len(x)
        acc /= float(cnt)
        print("Acc : {}".format(acc))

