import argparse
import os
import time
import sys
import cv2
import numpy as np
import CounterDetection
import CounterRecognition
import tensorflow as tf
import segmentation_models as sm
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
# research folder was created by the object detection API
path2scripts = r'\research'  # TODO: provide pass to the research folder
sys.path.insert(0, path2scripts)  # making scripts in models/research available for import
from object_detection.utils import label_map_util

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
    final = CounterDetection.detect_counter(preprocess_input, model, dir_img)
    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    prediction, img = CounterRecognition.recognize_digits(detect_fn, category_index, np.asarray(final), path2scripts)
    number = array_to_float(prediction, black, length)
    print('Done!')
    return number, img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--Name", help="Filename of Image to be processed")
    parser.add_argument("-w", "--Whole", help="Number of Whole Digits (ie. non-decimal digits)")
    parser.add_argument("-d", "--Digits", help="Number of Digits on the Meter Counter")
    args = parser.parse_args()

    if args.Output:
        print("Displaying Output as: % s" % args.Output)

    processing = True
    reading, processed_image = AMR(args.Name, args.Whole, args.Digits)
    print('Reading Value:', reading)
    plt.figure()
    plt.imshow(processed_image)
    outpath = os.path.join(os.getcwd(), args.Name.split('.')[0] + "_processed.jpg")
    plt.savefig(outpath)
    print('Output Image Saved at ', outpath)