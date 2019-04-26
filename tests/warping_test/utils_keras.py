from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.backend as K

import time

''''
    Differentiable depth image-based rendering utilities from:
    https://github.com/tinghuiz/SfMLearner
'''

def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
    TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = K.shape(z)[0]
    N = 1
    z = K.clip(z, -np.pi, np.pi)
    y = K.clip(y, -np.pi, np.pi)
    x = K.clip(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = K.expand_dims(K.expand_dims(z, -1), -1)
    y = K.expand_dims(K.expand_dims(y, -1), -1)
    x = K.expand_dims(K.expand_dims(x, -1), -1)

    zeros = K.zeros([B, N, 1, 1])
    ones  = K.ones([B, N, 1, 1])

    cosz = K.cos(z)
    sinz = K.sin(z)
    rotz_1 = K.concatenate([cosz, -sinz, zeros], axis=3)
    rotz_2 = K.concatenate([sinz,  cosz, zeros], axis=3)
    rotz_3 = K.concatenate([zeros, zeros, ones], axis=3)
    zmat = K.concatenate([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = K.cos(y)
    siny = K.sin(y)
    roty_1 = K.concatenate([cosy, zeros, siny], axis=3)
    roty_2 = K.concatenate([zeros, ones, zeros], axis=3)
    roty_3 = K.concatenate([-siny,zeros, cosy], axis=3)
    ymat = K.concatenate([roty_1, roty_2, roty_3], axis=2)

    cosx = K.cos(x)
    sinx = K.sin(x)
    rotx_1 = K.concatenate([ones, zeros, zeros], axis=3)
    rotx_2 = K.concatenate([zeros, cosx, -sinx], axis=3)
    rotx_3 = K.concatenate([zeros, sinx, cosx], axis=3)
    xmat = K.concatenate([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = K.batch_dot(K.batch_dot(xmat, ymat), zmat)
    return rotMat

def pose_vec2mat(vec):
    """Converts 6DoF parameters to transformation matrix
    Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
      A transformation matrix -- [B, 4, 4]
    """
    batch_size, _ = vec.get_shape().as_list()
    translation = K.slice(vec, [0, 0], [-1, 3])
    translation = K.expand_dims(translation, -1)
    rx = K.slice(vec, [0, 3], [-1, 1])
    ry = K.slice(vec, [0, 4], [-1, 1])
    rz = K.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = K.squeeze(rot_mat, axis=1)
    filler = K.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = K.tile(filler, [batch_size, 1, 1])
    transform_mat = K.concatenate([rot_mat, translation], axis=2)
    transform_mat = K.concatenate([transform_mat, filler], axis=1)
    return transform_mat

def pixel2cam(depth, pixel_coords, intrinsics, intrinsics_inverse=None, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    intrinsics_inverse: camera intrinsics inverse matrix (pre-computed) [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates
    Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    batch, height, width = depth.get_shape().as_list()
    depth = K.reshape(depth, [batch, 1, -1])
    pixel_coords = K.reshape(pixel_coords, [batch, 3, -1])
    if intrinsics_inverse is None:
        intrinsics_inverse = K.variable(np.linalg.inv(K.eval(intrinsics)))
    cam_coords = K.batch_dot(intrinsics_inverse, pixel_coords) * depth
    if is_homogeneous:
        ones = K.ones([batch, 1, height*width])
        cam_coords = K.concatenate([cam_coords, ones], axis=1)
    cam_coords = K.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords

def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
    Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.get_shape().as_list()
    cam_coords = K.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = K.batch_dot(proj, cam_coords)
    x_u = K.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = K.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = K.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = K.concatenate([x_n, y_n], axis=1)
    pixel_coords = K.reshape(pixel_coords, [batch, 2, height, width])
    return K.permute_dimensions(pixel_coords, pattern=[0, 2, 3, 1])

def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
    batch: batch size
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
    Returns:
    x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    #x_t = K.dot(K.ones(shape=K.stack([height, 1])),
    #              K.permute_dimensions(K.expand_dims(
    #                  K.variable(np.linspace(-1.0, 1.0, width)), 1), [1, 0]))
    #y_t = K.dot(K.expand_dims(K.variable(np.linspace(-1.0, 1.0, height)), 1),
    #              K.ones(shape=K.stack([1, width])))
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * K.cast(width - 1, 'float32')
    y_t = (y_t + 1.0) * 0.5 * K.cast(height - 1, 'float32')
    if is_homogeneous:
        ones = K.ones_like(x_t)
        coords = K.stack([x_t, y_t, ones], axis=0)
    else:
        coords = K.stack([x_t, y_t], axis=0)
    coords = K.tile(K.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords

def projective_inverse_warp(img, depth, pose, intrinsics):
    """Inverse warp a source image to the target image plane based on projection.

    Args:
    img: the source image [batch, height_s, width_s, 3]
    depth: depth map of the target image [batch, height_t, width_t]
    pose: target to source camera transformation matrix [batch, 6], in the
          order of tx, ty, tz, rx, ry, rz
    intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
    Source image inverse warped to the target image plane [batch, height_t,
    width_t, 3]
    """
    batch, height, width, _ = img.get_shape().as_list()
    # Convert pose vector to matrix
    pose = pose_vec2mat(pose)
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
    filler = K.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = K.tile(filler, [batch, 1, 1])
    intrinsics = K.concatenate([intrinsics, K.zeros([batch, 3, 1])], axis=2)
    intrinsics = K.concatenate([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source'
    # pixel frame.
    proj_tgt_cam_to_src_pixel = K.batch_dot(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    output_img = bilinear_sampler(img, src_pixel_coords)
    return output_img

def bilinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
    Returns:
    A new sampled image [batch, height_t, width_t, channels]
    """
    def _repeat(x, n_repeats):
        rep = K.permute_dimensions(
            K.expand_dims(K.ones(shape=K.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = K.cast(rep, 'float32')
        x = K.dot(K.reshape(x, (-1, 1)), rep)
        return K.reshape(x, [-1])

    coords_x = K.expand_dims(coords[:,:,:,0], axis=-1)
    coords_y = K.expand_dims(coords[:,:,:,1], axis=-1)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = K.cast(coords_x, 'float32')
    coords_y = K.cast(coords_y, 'float32')

    x0 = K.cast(K.cast(coords_x, 'int32'), 'float32')
    x1 = x0 + 1
    y0 = K.cast(K.cast(coords_y, 'int32'), 'float32')
    y1 = y0 + 1

    y_max = K.eval(K.cast([K.shape(imgs)[1] - 1], 'float32'))
    x_max = K.eval(K.cast([K.shape(imgs)[2] - 1], 'float32'))
    zero = K.eval(K.zeros([1], dtype='float32'))

    x0_safe = K.clip(x0, zero, x_max)
    y0_safe = K.clip(y0, zero, y_max)
    x1_safe = K.clip(x1, zero, x_max)
    y1_safe = K.clip(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * K.cast(K.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * K.cast(K.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * K.cast(K.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * K.cast(K.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from
    dim2 = K.cast(inp_size[2], 'float32')
    dim1 = K.cast(inp_size[2] * inp_size[1], 'float32')
    base = K.reshape(
        _repeat(
            K.cast(K.arange(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = K.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = K.reshape(imgs, K.stack([-1, inp_size[3]]))
    imgs_flat = K.cast(imgs_flat, 'float32')
    im00 = K.reshape(K.gather(imgs_flat, K.cast(idx00, 'int32')), out_size)
    im01 = K.reshape(K.gather(imgs_flat, K.cast(idx01, 'int32')), out_size)
    im10 = K.reshape(K.gather(imgs_flat, K.cast(idx10, 'int32')), out_size)
    im11 = K.reshape(K.gather(imgs_flat, K.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = w00*im00 + w01*im01 + w10*im10 + w11*im11
    return output


print '-- Keras:'
img_h = 4
img_w = 4
img_c = 3
batch_size = 5
decimal_precision = 5
benchmark_trials = 2

intrinsics_mat = np.array([[1.2, 0, 3.3], [2.8, 3.91, 1.02], [0, 5.8, 12.1]])
intrinsics = K.repeat_elements(K.expand_dims(K.variable(intrinsics_mat), axis=0), batch_size, 0)
intrinsics_inverse = K.repeat_elements(K.expand_dims(K.variable(np.linalg.inv(intrinsics_mat)), axis=0), batch_size, 0)

start_time = time.time()
for i in range(benchmark_trials):
    print '  Trial',(i+1)
    img = K.random_uniform_variable(shape=(batch_size,img_h,img_w,img_c), low=0, high=1, seed=i)
    depth = K.random_uniform_variable(shape=(batch_size,img_h,img_w), low=0, high=1, seed=i)
    pose = K.random_uniform_variable(shape=(batch_size,6), low=0, high=1, seed=i)
    result = K.eval(projective_inverse_warp(img, depth, pose, intrinsics))
    with np.printoptions(precision=decimal_precision, suppress=True):
        print result
print 'Time elapsed:', (time.time()-start_time)




