'''
Author: Wuyifan
Date: 2023-10-14 12:45:22
LastEditors: Wuyifan
LastEditTime: 2024-02-27 14:09:22
'''
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def weight_masked_softmax_cross_entropy(preds, labels, mask, weight):
    class_weights = tf.constant(weight, dtype=tf.float32)
    print(class_weights.shape)
    class_weights = tf.expand_dims(class_weights, 0)
    weights = tf.reduce_sum(class_weights * labels, axis = 1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    weighted_losses = unweighted_losses * weights
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    weighted_losses *= mask
    return tf.reduce_mean(weighted_losses)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))

    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def calculate_f_score(right, predict, class_num,weight):
    predict = list(predict)
    right = list(right)
    last = []
    result = []
    for i in range(class_num):
        result.append([0, 0, 0])
    assert len(right) == len(predict)
    last.append('precision')
    last.append('recall')
    last.append('F-measure')
    for i in range(len(right)):
        #print(right[i],predict[i])
        result[right[i]][1] += 1
        result[predict[i]][2] += 1
        if right[i] == predict[i]:
            result[predict[i]][0] += 1
    # f_score = []
    average_p = 0
    average_r = 0
    average_f_score = 0
    for i in range(class_num):
        p = result[i][0] / (result[i][2] or 1)

        r = result[i][0] / (result[i][1] or 1)
        f = (2*p*r) / ((p + r) or 1)
        average_p += p*weight[i]
        average_r += r*weight[i]
        average_f_score += f*weight[i]
        
        last.append(str(round(p, 4)))
        last.append(str(round(r, 4)))
        last.append(str(round(f, 4)))
    average_p = str(round(average_p , 4))
    average_r = str(round(average_r, 4))
    average_f_score = str(round(average_f_score, 4))
    last.append(average_p)
    last.append(average_r)
    last.append(average_f_score)
    return last


def calculate_weighted_f_score(right, predict, class_num,weight):
    f_score = calculate_f_score(right, predict, class_num)
    weighted_last = []
    for i in range(len(f_score)):
        weighted_last.append(float(f_score[i])*weight[i])
    return weighted_last