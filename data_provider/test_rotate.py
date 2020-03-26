import cv2
import numpy as np
from data_provider.data_utils import rotate_image
from data_provider.ICDAR_loader import ICDARLoader

def test():
    icdar_loader = ICDARLoader()
    img = cv2.imread("/Users/sunset/learn/dl/ch4_training_images/img_368.jpg")
    text_polygons, text_tags, labels = icdar_loader.load_annotation(gt_file="/Users/sunset/learn/dl/ch4_training_localization_transcription_gt/gt_img_368.txt")
    print(labels)
    img, text_polygons = rotate_image(img, text_polygons, 15)

    for poly in text_polygons:
        img = cv2.polylines(img, [poly.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

    cv2.imwrite("rotate.jpg", img)

if __name__ == '__main__':
    test()