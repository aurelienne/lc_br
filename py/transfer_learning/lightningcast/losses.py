# from https://github.com/karolzak/keras-unet/blob/master/keras_unet/losses.py

import tensorflow as tf
import tensorflow.keras.backend as K

# Tony provided this formula, which I think is equivalent
# K.mean((1.0 - q) * K.relu(e) + q * K.relu(-e), axis=-1)
def quantile_loss(q, y_true, y_pred):
    y_true = tf.cast(
        y_true, dtype=tf.float32
    )  # Convert y_true to float32 from uint8 to ensure compatibility and precision
    e = y_true - y_pred
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)


# From https://arxiv.org/pdf/2106.09757.pdf
def csi(
    use_as_loss_function,
    use_soft_discretization,
    hard_discretization_threshold=None,
    name="csi",
    index=0,
):
    def loss(target_tensor, prediction_tensor):

        target_shape = target_tensor.shape
        if target_shape[-1] < 9:
            # target_shape[-1] is number of outputs. LCv1 squeezed this output (i.e., doesn't exist)
            # So, we only enter here if we have < 9 outputs. For LCv1, this is dx (should always be > 9)
            # We do this in case there is more than one output
            target_tensor = target_tensor[..., index]
            prediction_tensor = prediction_tensor[..., index]

        if hard_discretization_threshold is not None:
            prediction_tensor = tf.where(
                prediction_tensor >= hard_discretization_threshold, 1.0, 0.0
            )
        elif use_soft_discretization:
            prediction_tensor = K.sigmoid(prediction_tensor)

        target_tensor = tf.cast(target_tensor, tf.float32)
        num_true_positives = K.sum(target_tensor * prediction_tensor)
        num_false_positives = K.sum((1 - target_tensor) * prediction_tensor)
        num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))

        denominator = (
            num_true_positives + num_false_positives + num_false_negatives + K.epsilon()
        )
        csi_value = num_true_positives / denominator

        if use_as_loss_function:
            return 1.0 - csi_value  # -tf.math.log(csi_value) #1. - csi_value
        else:
            return csi_value

    loss.__name__ = name

    return loss


def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# weighted binary cross-entropy
def wbce(
    y_true, y_pred, weight1=6.73785, weight0=0.5401
):  # weight1=15.501, weight0=0.5167) :
    y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    logloss = -(
        y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0
    )
    return K.mean(logloss, axis=-1)
