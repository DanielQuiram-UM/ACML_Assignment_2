import numpy as np


def color2ycrcb(arr):
    """
    Formulas are from:
    https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    This function will convert all images at once,
    which sames time compared to the OpenCV implementation that can only process images 1 by 1.
    """
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299*R + 0.587*G + 0.114*B
    Cr = (R-gray)*0.713 + 0.5
    Cb = (B-gray)*0.564 + 0.5
    return np.stack([gray, Cb, Cr], axis=-1), gray


def ycrcb2color(arr):
    """
    Formulas are from:
    https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    This function will convert all images at once,
    which sames time compared to the OpenCV implementation that can only process images 1 by 1.
    """
    gray, Cb, Cr = arr[..., 0], arr[..., 1], arr[..., 2]
    R = gray + 1.403*(Cr-0.5)
    G = gray - 0.714*(Cr-0.5) - 0.344*(Cb-0.5)
    B = gray + 1.773*(Cb-0.5)
    return np.stack([R, G, B], axis=-1)
