from sparse_multiclf import *
import sys

NUM_FEATURES = 1048575
NUM_CLASSES = 80001
NUM_EPOCH = 1

def main():
    argv = sys.argv
    if len(argv) < 3:
        print("usage: python3 main.py [B] [R] [gpu_option] [tag] [start] [end]")
        return
    B = int(argv[1])
    R = int(argv[2])
    gpu_option = '0'
    if len(argv) >= 4:
        gpu_option = argv[3]

    tag=None

    if len(argv) >= 5:
        tag = argv[4]

    start = 0
    end = R
    if len(argv)>= 7:
        start = int(argv[5])
        end = int(argv[6])

    clf = MultiClassifier(R=R,
                          B=B,
                          num_features=NUM_FEATURES,
                          num_classes=NUM_CLASSES,
                          seed=0,
                          tag=tag,
                          save_path=None,
                          load_hash=False
                          )
    clf.train("/home/frankie/ltcb/data/tfrecords/b20/train.tfrecords", gpu_option=gpu_option,
                                                         num_epochs=NUM_EPOCH,
                                                        start=start, end=end)
    clf.evaluate("/home/frankie/ltcb/data/tfrecords/b20/test.tfrecords", gpu_option=gpu_option)
    sys.exit()

if __name__ == '__main__':
  main()
