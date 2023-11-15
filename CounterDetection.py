import cv2
import numpy as np
from deskew import determine_skew
import scipy


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


def make_prediction(preprocess_input, model, img):
    # Predict meter counter location
    img = img.reshape(-1, 224, 224, 3).astype('uint8')
    img = preprocess_input(img)
    mask = model.predict(img)
    return mask


def make_mask(img, mask):
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
    extract = make_mask(img, mask)
    extracted = extract_counter(extract)
    height, width, _ = extracted.shape

    if min(height, width) / max(height, width) >= 0.6:
        extract = rotate_image(extract, 45)
        img = rotate_image(img, 45)
        extracted = extract_counter(extract)
        height, width, _ = extracted.shape

        # rotate counter if its in portrait
    if height > width:
        extract = cv2.rotate(extract, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        extracted = extract_counter(extract, img, 1)

    grayscale = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    # rotate image to check if there's improvements
    rotate = rotate_image(extract, angle)
    rotated = extract_counter(rotate)

    if rotated.shape[0] >= extracted.shape[0]:
        rotated = extract_counter(extract, img, 1)
    else:
        img_withrot = rotate_image(img, angle)
        rotated = extract_counter(rotate, img_withrot, 1)

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


def extract_counter(new_mask, img=0, rotated=0):
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


def detect_counter(preprocess_input, model, img):
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
    mask = make_prediction(preprocess_input, model, img_small)

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