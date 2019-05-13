import tensorflow as tf
import keras.backend as K
import numpy as np



def ssim_keras(x, y):
    """
    Computes a differentiable structured image similarity measure
    Taken from:
        https://github.com/tensorflow/models/tree/master/research/struct2depth
    """
    c1 = 0.01**2  # As defined in SSIM to stabilize div. by small denominator.
    c2 = 0.03**2
    mu_x = K.pool2d(x, (3,3), (1,1), 'valid', pool_mode='avg')
    mu_y = K.pool2d(y, (3,3), (1,1), 'valid', pool_mode='avg')
    sigma_x = K.pool2d(x**2, (3,3), (1,1), 'valid', pool_mode='avg') - mu_x**2
    sigma_y = K.pool2d(y**2, (3,3), (1,1), 'valid', pool_mode='avg') - mu_y**2
    sigma_xy = (K.pool2d(x * y, (3,3), (1,1), 'valid', pool_mode='avg') 
                - mu_x * mu_y)
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return K.clip((1 - ssim) / 2, 0, 1)

def ssim(x, y):
    """
    Computes a differentiable structured image similarity measure
    Taken from:
        https://github.com/tensorflow/models/tree/master/research/struct2depth
    """
    c1 = 0.01**2  # As defined in SSIM to stabilize div. by small denominator.
    c2 = 0.03**2
    mu_x = tf.contrib.slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = tf.contrib.slim.avg_pool2d(y, 3, 1, 'VALID')
    sigma_x = tf.contrib.slim.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
    sigma_y = tf.contrib.slim.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
    sigma_xy = tf.contrib.slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return tf.clip_by_value((1 - ssim) / 2, 0, 1)


x = np.random.rand(1,16,12,3)
y = np.random.rand(1,16,12,3)
x_tensor = K.variable(x)
y_tensor = K.variable(y)

print('Sum of tensors:')
print('[TF SSIM]:', K.eval(K.sum(ssim(x_tensor, y_tensor))))
print('[Keras SSIM]:', K.eval(K.sum(ssim_keras(x_tensor, y_tensor))))
print('[Difference]:',
      K.eval(K.sum(ssim(x_tensor, y_tensor) - ssim_keras(x_tensor, y_tensor))))


