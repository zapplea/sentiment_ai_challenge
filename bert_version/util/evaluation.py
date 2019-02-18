import numpy as np

def attr_evaluate(attr_pred, attr_label):
    """
    
    :param attr_pred: ( ,attr)
    :param senti_pred: ( ,attr,3)
    :param attr_label: 
    :param senti_label: 
    :return: 
    """

    length = attr_pred.shape[1]
    attr_class_f1 = []
    for i in range(length):
        attr_class_f1.append(attr_f1(attr_pred[:,i], attr_label[:,i]))

    attr_f1_res = np.mean(attr_class_f1)

    return attr_f1_res, attr_class_f1


def senti_evaluate(attr_pred, attr_label, senti_pred, senti_label):
    """
    :param attr_pred: ( , attr)
    :param attr_label: ( , attr)
    :param senti_pred: ( , attr, 3)
    :param senti_label: ( , attr, 3)
    :return: 
    """

    senti_pred = convert_to_one_hot(attr_pred * (senti_pred + 1), 4)
    senti_label = np.argmax(senti_label, axis=-1)
    senti_label = convert_to_one_hot(attr_label * (senti_label + 1), 4)

    length = attr_pred.shape[1]
    senti_class_f1 = []
    for i in range(length):
        senti_class_f1.append(senti_f1(senti_pred[:, i], senti_label[:, i]))

    senti_res = np.mean(senti_class_f1)

    return senti_res, senti_class_f1

def attr_f1(pred, label, epsilon = 1e-10):

    TP = np.sum(pred * label, axis=0)
    FP = np.sum(pred * (1-label), axis=0)
    FN = np.sum((1-pred) * label, axis=0)

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = np.mean(2 * precision * recall / (precision + recall + epsilon))

    return f1

def senti_f1(pred, label, epsilon = 1e-10):

    # TP:(4,)
    TP = np.sum(pred * label, axis=0)
    FP = np.sum(pred * (1-label), axis=0)
    FN = np.sum((1-pred) * label, axis=0)

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = np.mean(2 * precision * recall / (precision + recall + epsilon))

    return f1


def softmax(x, axis):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x, axis=axis)
    exp_x = np.exp(x, axis=axis)
    softmax_x = exp_x / np.sum(exp_x, axis=axis)
    return softmax_x

def convert_to_one_hot(y, C):
    y = np.int32(y)
    return np.eye(C)[y.reshape(-1)].reshape([-1,20,4])
