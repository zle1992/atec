
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        def score(label, pred, gate=0.5):

            if len(label.shape) == 1:
                p = (pred > gate).astype("int")
                p = np.squeeze(p)
                l = label
            else:
                p = np.argmax(pred, axis=1)
                l = np.argmax(label, axis=1)
            pre_score = precision_score(l, p, pos_label=1, average='binary')
            rec_score = recall_score(l, p, pos_label=1, average='binary')
            f_score = f1_score(l, p)
            return pre_score, rec_score, f_score
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = score(val_targ, val_predict)[2]
        _val_recall = score(val_targ, val_predict)[1]
        _val_precision = score(val_targ, val_predict)[0]
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return

# model.fit(X[train], Y[train], epochs=150, batch_size=batch_size,
#                       verbose=0, validation_data=(X[test], Y[test]),
#                       callbacks=[metrics])