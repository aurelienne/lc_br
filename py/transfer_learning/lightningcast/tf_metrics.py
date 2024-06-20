import tensorflow as tf
import tensorflow.keras.backend as K


def pod(
    use_soft_discretization, hard_discretization_threshold=None, name="pod", index=0
):
    def pod(target_tensor, prediction_tensor):

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
        num_false_negatives = K.sum(target_tensor * (1 - prediction_tensor))

        denominator = num_true_positives + num_false_negatives + K.epsilon()
        pod_value = num_true_positives / denominator

        return pod_value

    pod.__name__ = name

    return pod


# ------------------------------------------------------------------------------------------------


def far(
    use_soft_discretization, hard_discretization_threshold=None, name="far", index=0
):
    def far(target_tensor, prediction_tensor):

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

        denominator = num_true_positives + num_false_positives + K.epsilon()
        far_value = num_false_positives / denominator

        return far_value

    far.__name__ = name

    return far


# ------------------------------------------------------------------------------------------------


# Observed counts. The numerator for reliability.
def obs_ct(threshold1, threshold2, name='obsct', index=0):
    def obs_ct(target_tensor, prediction_tensor):
     
        target_shape = target_tensor.shape
        if target_shape[-1] < 9:
            # target_shape[-1] is number of outputs. LCv1 squeezed this output (i.e., doesn't exist)
            # So, we only enter here if we have < 9 outputs. For LCv1, this is nx (should always be > 9)
            # We do this in case there is more than one output
            target_tensor = target_tensor[..., index]
            prediction_tensor = prediction_tensor[..., index]

        condition = tf.logical_and(prediction_tensor >= threshold1, prediction_tensor < threshold2)
    
        prediction_tensor = tf.where(condition, 1., 0.)
        target_tensor = tf.cast(target_tensor, tf.float32)
    
        num_hits = K.sum(target_tensor * prediction_tensor)
    
        return num_hits
  
    obs_ct.__name__ = name
  
    return obs_ct


#------------------------------------------------------------------------------------------------


# Forecast counts. The denomenator for reliability.
def fcst_ct(threshold1, threshold2, name='fcstct', index=0):
    def fcst_ct(target_tensor, prediction_tensor):
  
        target_shape = target_tensor.shape
        if target_shape[-1] < 9:
            # target_shape[-1] is number of outputs. LCv1 squeezed this output (i.e., doesn't exist)
            # So, we only enter here if we have < 9 outputs. For LCv1, this is nx (should always be > 9)
            # We do this in case there is more than one output
            target_tensor = target_tensor[..., index]
            prediction_tensor = prediction_tensor[..., index]

        condition = tf.logical_and(prediction_tensor >= threshold1, prediction_tensor < threshold2)
    
        prediction_tensor = tf.where(condition, 1., 0.)
    
        num_valid_preds = K.sum(prediction_tensor)
    
        return num_valid_preds
  
    fcst_ct.__name__ = name
  
    return fcst_ct


#------------------------------------------------------------------------------------------------
