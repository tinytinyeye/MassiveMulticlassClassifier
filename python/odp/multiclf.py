import tensorflow as tf
from hash import *
from clf import *

class MultiClassifier(object):

    def __init__(self, R, B, p=105943):
        """
        M - list of trained models
        R - number of models
        B - number of classes in each model
        p - prime number used in hashing
        a - parameters used in hashing
        b - parameters used in hashing
        """
        self.M = None
        self.W = None
        self.R = R
        self.B = B
        self.p = p
        self.a = []
        self.b = []

    def recover(self, a, b, p, B, R, M, W):
        self.a = a
        self.b = b
        self.p = p
        self.R = R
        self.B = B
        self.M = M
        self.W = W

    def get_trained_models(self):
        return self.M

    def get_weight(self):
        return self.W

    def train(self, X, y, n_epoch=100):
        """
        train models for the multi classifier

        params:
        X - train data, numpy array, [num_train * num_features]
        y - train label, numpy array, [num_train * 1]
        num_epoch - epoches to train
        """
        if self.M is None:
            M = []
            a, b, y_hash = hash_and_test(y, self.R, self.B, self.p)
            self.a = a
            self.b = b
            for i in range(self.R):
                m = single_classifier_train(X, y_hash[i], num_epoch=n_epoch)
                M.append(m)
                print("model %d taining finished" % (i))
            self.M = M
        else:
            y_hash = hash_vector(y, self.a, self.b, self.B, self.p)
            for i in range(self.R):
                self.M[i].fit(X, y_hash[i], n_epoch=n_epoch, show_metric=True)
                print("model %d taining finished" % (i))
        print("train finished")
    def get_sub_probas(self, X):
        """
        get probabilities for each sub classifier

        params:
        X - predict data

        returns:
        P - a list of probabilities, P[i] is the probabilities get from
        sub-classifier i, with dimension [num_test * B]
        """
        P = []
        for i in range(self.R):
            probs = self.M[i].predict(X)
            P.append(probs) # P[i] is (n_test * B)
        return P;

    def predict_with_average(self, X, k, top_k=1):
        """
        get prediction for each sample in X, using average to aggregate results
        from all sub-classifiers

        params:
        X - predict data
        k - number of distinct labels
        top_k - save top k predictions for return

        returns:
        y_pred - prediction labels for each sample
        """
        num_test = X.shape[0]
        GP = np.zeros((num_test, k)) # global probabilities for each class
        P = []
        for i in range(self.R):
            probs = self.M[i].predict(X)
            P.append(probs) # P[i] is (n_test * B)

        for i in range(k):
                # aggregate with average
                # r is a matrix that contains result of the following:
                # for class i in classifier j, the probability for class
                # h_j(i)
                probs_i = np.zeros((num_test, self.R))
                for j in range(self.R):
                    probs_i[:, j] = P[j][:, H(i, self.a[j], self.b[j], self.B, self.p)]
                GP[:, i] = np.mean(probs_i, axis=1)

        if (top_k <= 1):
            y_pred = np.argmax(GP, axis = 1)
        else:
            y_pred = np.argsort(GP, axis = 1)[:, -top_k:]
        return y_pred

    def predict_with_weight(self, X, k, top_k=1):
        """
        get prediction for each sample in X, using average to aggregate results
        from all sub-classifiers

        params:
        X - predict data
        k - number of distinct labels
        top_k - save top k predictions for return

        returns:
        y_pred - prediction labels for each sample
        """
        num_test = X.shape[0]
        GP = np.zeros((num_test, k)) # global probabilities for each class
        P = []
        for i in range(self.R):
            probs = self.M[i].predict(X)
            P.append(probs) # P[i] is (n_test * B)

        for i in range(k):
                # aggregate with average
                # r is a matrix that contains result of the following:
                # for class i in classifier j, the probability for class
                # h_j(i)
                probs_i = np.zeros((num_test, self.R))
                for j in range(self.R):
                    probs_i[:, j] = P[j][:, H(i, self.a[j], self.b[j], self.B, self.p)]
                # GP[:, i] = np.mean(probs_i, axis=1)
                GP[:, i] = np.dot(probs_i, np.array(self.W))

        if (top_k <= 1):
            y_pred = np.argmax(GP, axis = 1)
        else:
            y_pred = np.argsort(GP, axis = 1)[:, -top_k:]
        return y_pred

    def fine_tune(self, X_data, y_data, num_epoch=100, top_k=1):
        num_test = X_data.shape[0]
        batch_size = 600
        k = len(np.unique(y_data))
        P_list = []
        for i in range(self.R):
            probs = self.M[i].predict(X_data)
            P_list.append(probs) # P[i] is (n_test * B)
        P = np.dstack(P_list)
        # used as input data
        P = np.rollaxis(P, -1) # P is now (R * num_test * B)

        # input
        X = tf.placeholder(tf.float32, shape=([self.R, None, self.B]))
        y = tf.placeholder(tf.int32, shape=([None]))
        # weight
        # init_weight = [1 / self.R] * self.R
        # init_weight = np.array(init_weight)
        # weights = tf.Variable(init_weight, name="weights", dtype=tf.float32)
        # weights = tf.Variable(tf.random_uniform(
        #                             [self.R],
        #                             minval=0.0,
        #                             maxval=1.0,
        #                             dtype=tf.float32,
        #                             name="weights"
        #                         ))
        weights = tf.Variable(tf.truncated_normal([self.R], stddev=0.1), name="weights")

        def loss_func(X_train, y_train, w):
            GP_list = []
            for i in range(k):
                probs_i_list = []
                for j in range(self.R):
                    probs_i_list.append(X_train[j][:, H(i, self.a[j], self.b[j], self.B, self.p)])
                probs_i = tf.stack(probs_i_list, axis=1)
                GP_i = tf.matmul(probs_i, tf.expand_dims(w, 1))
                GP_list.append(GP_i)
            GP = tf.concat(GP_list, axis=1)
            # y_one_hot = tf.one_hot(y_train, k, on_value=1.0, off_value=0.0, axis=-1)
            # return tf.nn.softmax_cross_entropy_with_logits(logits=GP, labels=y_one_hot)
            return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=GP, labels=y_train)

        loss = tf.reduce_mean(loss_func(X, y, weights))
        train = tf.train.AdamOptimizer().minimize(loss)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init) # reset values to wrong

        for i in range(num_epoch):
            # printProgressBar(i, num_epoch, prefix = 'Progress:', suffix = 'Complete', length = 50)
            for j in range(num_test // batch_size):
                x_batch = P[:, j*batch_size:(j+1)*batch_size, :]
                y_batch = y_data[j*batch_size:(j+1)*batch_size]
                sess.run(train, {X:x_batch, y:y_batch})
                # sess.run(train, {X:P, y:y_data})
            curr_W, curr_loss = sess.run([weights, loss], {X:P, y:y_data})
            print("W: %s loss: %s"%(curr_W, curr_loss))

        curr_W, curr_loss = sess.run([weights, loss], {X:P, y:y_data})
        print("W: %s loss: %s"%(curr_W, curr_loss))
        self.W = curr_W
        return curr_W

    @staticmethod
    def calc_accuracy(y_pred, y_test, top_k=False):
        """
        compare prediction and true labels for test data to
        get accuracy value.

        params:
        y_pred - prediction labels
        y_test - true labels
        top_k - if top_k is enabled

        returns:
        test_accuracy - accuracy for the test data
        """
        test_accuracy = 0
        if (not top_k):
            test_accuracy = np.mean(y_test == y_pred)
        else:
            correct = 0
            total = y_test.shape[0]
            for i in range(total):
                if y_test[i] in y_pred[i]:
                    correct += 1
            print("{}/{}".format(correct, total))
            test_accuracy = correct / total
        return test_accuracy
