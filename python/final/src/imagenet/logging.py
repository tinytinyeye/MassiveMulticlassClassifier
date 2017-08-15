import os
import Timer from util

class Log(object):

    def __init__(id_, name, B, R):
        self.id = id_
        self.name = name
        self.B = B
        self.R = R
        self.train_log = None
        self.test_log = "./" + "_".join(name, id_, str(B), str(R), "predict") + ".log"

    def init_train_log():
        train_log_path = "./" + "_".join(name, id_, str(B), str(R), "train") + ".log"
