import cv2
import numpy as np


def resize(img, height=300):
    rat = height / img.shape[0]
    return cv2.resize(img, (int(rat * img.shape[1]), height))


def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def find_item_2(img_path):
    src = cv2.imread(img_path)
    src = resize(src)

    hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(src, src, mask=mask)
    b, g, r = cv2.split(result)
    g = clahe(g, 5, (3, 3))

    # Adaptive Thresholding to isolate the bed
    img_blur = cv2.blur(g, (9, 9))
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 2)

    im, contours, hierarchy = cv2.findContours(img_th,
                                               cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

    # Filter the rectangle by choosing only the big ones
    # and choose the brightest rectangle as the bed
    max_brightness = 0
    canvas = src.copy()
    brightest_rectangle = None
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w * h > 40000:
            mask = np.zeros(src.shape, np.uint8)
            mask[y:y + h, x:x + w] = src[y:y + h, x:x + w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness
            cv2.drawContours(canvas, [cnt], -1, (0, 255, 0), 2)
            cv2.imshow("mask", mask)
            cv2.imshow("contour", canvas)
            cv2.waitKey(0)
    if brightest_rectangle is not None:
        x, y, w, h = brightest_rectangle
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("canvas", canvas)
        cv2.waitKey(0)


def find_main_item(img_path):
    img = cv2.imread(img_path)
    img = resize(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 0, 80), (255, 255, 255))  # set to the color you want to detect

    blur = cv2.blur(mask, (5, 5))

    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    cnts = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    cv2.imshow('blur', blur)
    cv2.imshow('thresh', thresh)
    cv2.imshow('cnts', cnts)

    cv2.waitKey(0)


def detect_edges(img_path):
    img = cv2.imread(img_path)
    img = resize(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh_gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
    #                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 150)))

    edged = cv2.Canny(gray, 50, 150)
    edged = cv2.dilate(edged, None, iterations=3)
    edged = cv2.erode(edged, None, iterations=3)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for (i, c) in enumerate(cnts):
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue
        # compute the rotated bounding box of the contour, then
        # draw the contours
        box = cv2.minAreaRect(c) # x, y, w, h
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        # show the original coordinates
        print("Object #{}:".format(i + 1))
        print(box)
    cv2.imshow('edged', edged)
    cv2.imshow('thresh_gray', gray)
    cv2.imshow('contour', img)
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_path = '../big_transfer/images/clock.jpg'

    detect_edges(img_path)
