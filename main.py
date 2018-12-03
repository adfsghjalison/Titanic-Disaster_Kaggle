from flags import FLAGS
from model import DNN


def run():
  model = DNN(FLAGS)
  if FLAGS.mode == 'train':
    model.train()
  elif FLAGS.mode == 'test':
    model.test()
  elif FLAGS.mode == 'val':
    model.val()

if __name__ == '__main__':
  run()

