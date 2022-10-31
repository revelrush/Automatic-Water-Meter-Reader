import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from deskew import determine_skew
import scipy
import segmentation_models as sm
import os
import time
import sys
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)

# research folder was created by the object detection API
path2scripts = r'\research'  # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts)  # making scripts in models/research available for import
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

print('Loading models...', end='')
start_time = time.time()

# Create image segmentation model
sm.set_framework('tf.keras')
sm.framework()
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    opt,
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# Change to path of the image segmentation weights
fname = 'weights-file.hdf5'  # TODO: provide  location to the indicated file
model.load_weights(fname)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

# Change to path of the object detection weights and the label file
PATH_TO_SAVED_MODEL = r'my_faster_rcnn_resnet101-complete\saved_model'  # TODO: provide pass to the research folder, feel free to use the 50 or 101 resnet variant
PATH_TO_LABELS = 'label_map.pbtxt'  # TODO: provide pass to the research folder

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


def crop(img, bg, mask) -> np.array:
    # Function takes image, background, and mask, and crops the image, the cropped image should correspond only with
    # the positive portion of the mask.
    fg = cv2.bitwise_or(img, img, mask=mask)
    fg_back_inv = cv2.bitwise_or(bg, bg, mask=cv2.bitwise_not(mask))
    New_image = cv2.bitwise_or(fg, fg_back_inv)
    return New_image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def MakePrediction(img):
    # Predict meter counter location
    img = img.reshape(-1, 224, 224, 3).astype('uint8')
    img = preprocess_input(img)
    mask = model.predict(img)
    return mask


def MakeMask(img, mask):
    # Overlay prediction on image to create mask for cropping
    img = img.astype('uint8')
    mask = mask.astype('uint8')
    mask = np.asarray(mask)
    h, w, _ = img.shape
    mask = cv2.resize(mask, (w, h))  # Resize image
    # Create background array.
    bg = np.zeros_like(img, 'uint8')
    new_mask = crop(img, bg, mask)
    new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2RGB)
    return new_mask


def rotation_checker(img, mask):
    # extract counter with resized mask w/o rotation
    extract = MakeMask(img, mask)
    extracted = ExtractCounter(extract)
    height, width, _ = extracted.shape

    if min(height, width) / max(height, width) >= 0.6:
        extract = rotate_image(extract, 45)
        img = rotate_image(img, 45)
        extracted = ExtractCounter(extract)
        height, width, _ = extracted.shape

        # rotate counter if its in portrait
    if height > width:
        extract = cv2.rotate(extract, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        extracted = ExtractCounter(extract, img, 1)

    grayscale = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    # rotate image to check if there's improvements
    rotate = rotate_image(extract, angle)
    rotated = ExtractCounter(rotate)

    if rotated.shape[0] >= extracted.shape[0]:
        rotated = ExtractCounter(extract, img, 1)
    else:
        img_withrot = rotate_image(img, angle)
        rotated = ExtractCounter(rotate, img_withrot, 1)

    return rotated



def resize_input(img_link, option=0):
    # Function to resize image, resizes large images to when option 1 is chosen for optimization
    # if option 0 is taken, it creates a 224 by 224 pixel copy of the image for the image segmentation model
    img = cv2.imread(img_link)
    height, width, _ = img.shape
    if option == 0:
        img = cv2.resize(img, (224, 224))
    else:
        if height >= width and width > 1200:
            scale_percent = 1200 / width
        elif width > height and width > 1500:
            scale_percent = 1500 / width
        else:
            scale_percent = 1
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img.astype('uint8')


def ExtractCounter(new_mask, img=0, rotated=0):
    # Extract portion of image where meter reading is.
    # Use min and max x and y coordinates to obtain final image.
    where = np.array(np.where(new_mask))
    x1, y1, z1 = np.amin(where, axis=1)
    x2, y2, z2 = np.amax(where, axis=1)
    if rotated == 1:
        sub_image = img.astype('uint8')[x1:x2, y1:y2]
    else:
        sub_image = new_mask.astype('uint8')[x1:x2, y1:y2]
    return sub_image


def CounterDetector(img):
    # Function for the actual counter detection stage
    # extract images for two sizes, one for reading, the other to overlay on
    img_small = resize_input(img)
    input_img = resize_input(img, 1)
    # Get dimensions
    input_w = int(input_img.shape[1])
    input_h = int(input_img.shape[0])
    dim = (input_w, input_h)

    # only denoise if image size is big
    if max(input_w, input_h) > 4000:
        img_small = cv2.fastNlMeansDenoisingColored(img_small, None, 10, 10, 7, 21)

    # Load model, preprocess input, and obtain prediction.
    mask = MakePrediction(img_small)

    # fill holes to make solid mask
    mask = mask.astype('uint8')
    mask = scipy.ndimage.morphology.binary_fill_holes(mask[0, :, :, 0]).astype('uint8')

    # Resize mask to equal input image size.
    mask = cv2.resize(mask, dsize=dim, interpolation=cv2.INTER_AREA)

    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((10, 10), np.uint8)
    # thin out boundaries
    mask = cv2.dilate(mask, kernel, iterations=3)

    # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    # extract rotated image as a whole
    rotated = rotation_checker(input_img, mask)

    if rotated.shape[1] > 640:
        scale_percent = 640 / rotated.shape[1]
        width = int(rotated.shape[1] * scale_percent)
        height = int(rotated.shape[0] * scale_percent)
        dim = (width, height)
        rotated = cv2.resize(rotated, dim, interpolation=cv2.INTER_AREA)
    # plt.imshow(rotated)
    # plt.show()
    return rotated


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


def shortlist(boxes, number, score, th=0.7):
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


def inference(image_np):
    # The actual function for the counter prediction stage
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
    boxes, number, score = shortlist(boxes, number, score)

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


def array_to_float(array, black=5, length=8):
    # converts the prediction results from an array of digits to the decimal format
    number = ''.join(map(str, array))
    if len(array) >= length:
        if length == black:
            number = float(number)
        else:
            number = float(number[:black] + '.' + number[black:length])
    elif len(array) > black:
        number = float(number[:black] + '.' + number[black:])
    elif number == '':
        number = 0
    else:
        number = float(number)
    return number


def AMR(dir_img, black, length):
    print('Now Reading Image......')
    final = CounterDetector(dir_img)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    prediction, img = inference(np.asarray(final))
    number = array_to_float(prediction, black, length)
    print('Done!')
    return number, img


processing = True
while processing:
    filename = str(input('Filename of Image to be processed:'))
    length = int(input('Number of Digits on the Meter Counter'))
    black = int(input('Number of Whole Digits (ie. non-decimal digits):'))
    reading, processed_image = AMR(filename, length, black)
    print('Reading Value:', reading)
    plt.figure()
    plt.imshow(processed_image)
    plt.savefig(os.path.join(os.getcwd(), filename.split('.')[0] + "_processed.jpg"))
    prompt = str(input('Would you like to try again [Y/N]?'))
    if prompt != 'Y':
        break
