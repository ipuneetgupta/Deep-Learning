import cv2
import numpy as np 
import random
import cv2
import os
from scipy.ndimage import convolve
import imutils
from skimage.filters import threshold_adaptive

def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return None, 0

def DigitAugmentation(frame, dim = 32):
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    random_num = np.random.randint(0,9)

    if (random_num % 2 == 0):
        frame = noise_add(frame)
    if(random_num % 3 == 0):
        frame = pixel_enlarge(frame)
    if(random_num % 2 == 0):
        frame = stretch(frame)
    frame = cv2.resize(frame, (dim, dim), interpolation = cv2.INTER_AREA)

    return frame 

def noise_add(image):
    prob = random.uniform(0.01, 0.05)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noise_img = image.copy()
    noise_img[rnd < prob] = 0
    noise_img[rnd > 1 - prob] = 1
    return noise_img

def pixel_enlarge(image):
    dim = np.random.randint(8,12)
    image = cv2.resize(image, (dim, dim), interpolation = cv2.INTER_AREA)
    image = cv2.resize(image, (16, 16), interpolation = cv2.INTER_AREA)
    return image

def stretch(image):
    r = np.random.randint(0,3)*2
    if np.random.randint(0,2) == 0:
        frame = cv2.resize(image, (32, r+32), interpolation = cv2.INTER_AREA)
        return frame[int(r/2):int(r+32)-int(r/2), 0:32]
    else:
        frame = cv2.resize(image, (r+32, 32), interpolation = cv2.INTER_AREA)
        return frame[0:32, int(r/2):int(r+32)-int(r/2)]
    
def pre_process(image, inverse = False):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray_image = image
        pass
    
    if inverse == False:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(th2, (32,32), interpolation = cv2.INTER_AREA)
    return resized

def order_points(pts):
    rectangle = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rectangle[0] = pts[np.argmin(s)]
    rectangle[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rectangle[1] = pts[np.argmin(diff)]
    rectangle[3] = pts[np.argmax(diff)]

    return rectangle

def four_point_transform(image, pts):
    rectangle = order_points(pts)
    (tl, tr, br, bl) = rectangle

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

 
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rectangle, dst)
    img_trans = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return img_trans

def creditcard_Scan(image):
    orig_height, orig_width = image.shape[:2]
    ratio = image.shape[0] / 500.0

    orig = image.copy()
    image = imutils.resize(image, height = 500)
    orig_height, orig_width = image.shape[:2]
    Original_Area = orig_height * orig_width
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
   
    _, contours, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    
    i = 0
    for c in contours:

        i+=1
        area = cv2.contourArea(c)
        if area < (Original_Area/3):
            print("Image Invalid  -> "+str(i)+str(c.shape))
            return("image has not enough area")
        peri = cv2.arcLength(c, True)
        epsilon = 0.02 * peri
        approx = cv2.approxPolyDP(c, epsilon, True)

     
        if len(approx) == 4:
            screenCnt = approx
            break

   
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)

    transform_image = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    cv2.resize(transform_image, (640,403), interpolation = cv2.INTER_AREA)
    cv2.imwrite("credit_card_transform.jpg", transform_image)
    transform_image = cv2.cvtColor(transform_image, cv2.COLOR_BGR2GRAY)
    transform_image = transform_image.astype("uint8") * 255
    cv2.imshow("Extracted Credit Card", transform_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return transform_image

def cordinate_cntr_xaxis(contours):
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))
    else:
        pass