import tensorflow as tf

class Score:

    def __int__(self):
        pass

    def f1Score(self, y_true, y_pred, epsilon=1e-7):
        y_pred = tf.round(y_pred)

        TP = tf.reduce_sum(tf.cast(y_pred * y_true, 'float'), axis=0)
        FP = tf.reduce_sum(tf.cast((1 - y_pred) * y_true, 'float'), axis=0)
        FN = tf.reduce_sum(tf.cast(y_pred * (1 - y_true), 'float'), axis=0)

        P = TP / (TP + FP + epsilon)
        R = TP / (TP + FN + epsilon)

        F1 = 2 * P * R / (P + R + epsilon)
        F1 = tf.where(tf.is_nan(F1), tf.zeros_like(F1), F1)
        return F1

    def weightsF1Score(self, y_true, y_pred):
        label = tf.argmax(y_true)
        score = 0.0

        if label == 0:
            score += 0.2 * self.f1Score(y_true, y_pred)
        elif label == 1:
            score += 0.2 * self.f1Score(y_true, y_pred)
        elif label == 2:
            score += 0.6 * self.f1Score(y_true, y_pred)
        else:
            print("no exist label:", label)
        return score
