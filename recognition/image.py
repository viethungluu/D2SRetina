import numpy as np
import cv2

def compute_resize_scale(image_shape, target_size=224):
    """ Compute an image scale such that the image size is constrained to target_size x target_size.

    Args
        target_size: The image's side
        If after resizing the image's side is below target_size, padding with zero.

    Returns
        A resizing scale.
    """

    (rows, cols, _) = image_shape
    largest_side = max(rows, cols)
    # rescale the image so the largest side is target_size
    scale = target_size / largest_side

    return scale

def resize_image(img, target_size=224):
    """ Resize an image such that the size is constrained to target_size x target_size.

    Args
        target_size: The image's side
        If after resizing the image's side is below target_size, padding with zero.

    Returns
        A resized image.
    """

    # compute scale to resize the image
    scale   = compute_resize_scale(img.shape, target_size=target_size)
    # resize the image with the computed scale
    img     = cv2.resize(img, None, fx=scale, fy=scale)

    # rescale image
    result  = np.zeros((target_size, target_size, img.shape[2]), dtype=np.uint8)
    result[: img.shape[0], : img.shape[1], ...]  = img

    return result, scale