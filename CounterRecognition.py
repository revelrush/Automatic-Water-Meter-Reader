import numpy as np
import tensorflow as tf
import sys
from PIL import Image


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def overlap_checker(box1, box2):
    # checks for overlaps between prediction results on the object detection model
    if box1[3] >= box2[1] and box2[3] >= box1[1]:
        return True
    else:
        return False


def redundant_eliminator(boxes, number, score):
    # removes overlap and excess predictions based on the confidence level
    count = len(number)
    rejects = []
    for i, digit in enumerate(number):
        for j in range(i + 1, count):
            result = overlap_checker(boxes[i], boxes[j])
            if result == True:
                if i in rejects or j in rejects:
                    pass
                elif score[i] < score[j]:
                    rejects.append(i)
                else:
                    rejects.append(j)
    boxes = np.delete(boxes, rejects, 0)
    number = np.delete(number, rejects, 0)
    score = np.delete(score, rejects, 0)
    return boxes, number, score


def digit_sorter(boxes, number, score):
    # sorts prediction results based on coordinates, from left to right
    n = len(number)
    for i in range(n):
        already_sorted = True
        for j in range(n - i - 1):
            if boxes[j][1] > boxes[j + 1][1]:
                temp = np.copy(boxes[j])
                boxes[j] = boxes[j + 1]
                boxes[j + 1] = temp
                number[j], number[j + 1] = number[j + 1], number[j]
                score[j], score[j + 1] = score[j + 1], score[j]
                already_sorted = False
        if already_sorted:
            break
    return boxes, number, score


def shortlist_predictions(boxes, number, score, th=0.7):
    # removes all predictions results below the desired threshold
    for i, rate in enumerate(score):
        if rate < th:
            break
    boxes = boxes[:i]
    number = number[:i]
    score = score[:i]
    boxes, number, score = redundant_eliminator(boxes, number, score)
    boxes, number, score = digit_sorter(boxes, number, score)
    return boxes, number, score


def recognize_digits(detect_fn, category_index, image_np, path2scripts):
    # The actual function for the counter prediction stage
    sys.path.insert(0, path2scripts)  # making scripts in models/research available for import
    from object_detection.utils import visualization_utils as viz_utils

    # The input needs to be a tensor
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an extra axis
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}

    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    boxes = detections['detection_boxes'].copy()
    number = detections['detection_classes'].copy()
    score = detections['detection_scores'].copy()
    boxes, number, score = shortlist_predictions(boxes, number, score)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        boxes,
        number,
        score,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)
    for i, digit in enumerate(number):
        if digit == 10:
            number[i] = 0
    return number, image_np_with_detections


