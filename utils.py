import cv2

import numpy as np


# class to rgb colour pallet
color_dict = {
    0: (0, 0, 0),        # BG
    1: (239, 164, 0),    # EX
    2: (0, 186, 127),    # HE
    3: (0, 185, 255),    # SE
    4: (34, 80, 242),    # MA
    5: (73, 73, 73),     # OD
    6: (255, 255, 255),  # VB
}


def rgb_to_onehot(rgb_arr, color_dict):
    """
    Converts a rgb label map to onehot label map defined by color_dict
        Parameters:
            rgb_arr (array): rgb label mask with shape (H x W x 3)
            color_dict (dict): dictionary mapping of class to colour
        Returns:
            arr (array): onehot label map of shape (H x W x n_classes)
    """
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(color_dict):
        arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) == color_dict[i], axis=1).reshape(shape[:2])
    return arr


def onehot_to_rgb(onehot_arr, color_dict):
    """
    Converts an onehot label map to rgb label map defined by color_dict
        Parameters:
            onehot_arr (array): onehot label mask with shape (H x W x n_classes)
            color_dict (dict): dictionary mapping of class to colour
        Returns:
            arr (array): rgb label map of shape (H x W x 3)
    """
    shape = onehot_arr.shape[:2]
    mask = np.argmax(onehot_arr, axis=-1)
    arr = np.zeros(shape+(3,), dtype=np.uint8)
    for i, cls in enumerate(color_dict):
        arr = arr + np.tile(color_dict[cls], shape + (1,)) * (mask[..., None] == cls)
    return arr


def fix_pred_label(labels):
    """
    Post-processing fixes for the prediction of VB and BG label class,
    the Vitrous Body should be consistently spherical on a black background
        Parameters:
            labels (tensor): A 4-D array of predicted label
              with shape (batch x H x W x 7)
        Returns:
            fixed_labels (array): shape (batch x H x W x 7)
    """
    shape = labels.shape[1:-1]
    VB = np.uint8(cv2.circle(np.zeros(shape), (shape[0]//2, shape[1]//2), min(shape) // 2, 1, -1))[..., None]
    BG = np.uint8(VB == 0)

    VB = VB - np.sum(labels[..., 1:-1], axis=-1)[..., None]
    BG = np.broadcast_to(BG, VB.shape)

    fixed_labels = np.concatenate([BG, labels[..., 1:-1], VB], axis=-1)

    return fixed_labels
