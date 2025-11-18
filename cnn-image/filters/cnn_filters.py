import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def apply_blur(image: Image.Image) -> Image.Image:
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    else:
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
    return Image.fromarray(img_array)


def apply_edge_detection(image: Image.Image) -> Image.Image:
    img_array = np.array(image)
    gray = (
        cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if len(img_array.shape) == 3
        else img_array
    )
    edges = cv2.Canny(gray, 100, 200)
    if len(img_array.shape) == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)


def apply_sharpen(image: Image.Image) -> Image.Image:
    img_array = np.array(image)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    if len(img_array.shape) == 3:
        sharpened = cv2.filter2D(img_array, -1, kernel)
    else:
        sharpened = cv2.filter2D(img_array, -1, kernel)
    return Image.fromarray(sharpened)


def apply_filters(image: Image.Image) -> tuple:
    blur_img = apply_blur(image)
    edge_img = apply_edge_detection(image)
    sharp_img = apply_sharpen(image)
    return blur_img, edge_img, sharp_img
