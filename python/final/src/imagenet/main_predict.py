from dense_multiclf import *
import sys

NUM_FEATURES = 6144
NUM_CLASSES = 21841

def main():
    """
    main program to predict, used when two trainings are executed in parallel
    """
    argv = sys.argv
    if len(argv) < 4:
        print("usage: python3 main.py [B] [R] [tag] [gpu_option] [start] [end]")
        return
    B = int(argv[1])
    R = int(argv[2])
    tag = argv[3]
    gpu_option = '0'
    if len(argv) >= 5:
        gpu_option = argv[4]
    start = 0
    end = R
    if len(argv) >= 7:
        start = int(argv[5])
        end = int(argv[6])
    clf = MultiClassifier(R=R,
                          B=B,
                          num_features=NUM_FEATURES,
                          num_classes=NUM_CLASSES,
                          seed=0,
                          tag=tag,
                          save_path=None
                          )

    # some probs are not computed
    if start >= 0:
        clf.predict("./data/tfrecords/testing.tfrecords",
                    gpu_option=gpu_option,
                    start=start,
                    end=end)
        clf.evaluate("./data/tfrecords/testing.tfrecords",
                    gpu_option=gpu_option,
                    load_probs=True)
    # y_test = np.load("./data/y_test.npy")
    # clf.evaluate(y_test)
    else:
        clf.evaluate("./data/tfrecords/testing.tfrecords", gpu_option=gpu_option, load_probs=True)
    sys.exit()

if __name__ == '__main__':
  main()
