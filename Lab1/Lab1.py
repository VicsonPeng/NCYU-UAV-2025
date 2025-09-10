import cv2
import numpy as np
import math

def grayscaleColorFilter(imagePath):
    img = cv2.imread(imagePath)
    result = img.copy()
    height, width = img.shape[:2]

    for i in range(height):
        for j in range(width):
            b, g, r = img[i, j]
            if b > 100 and b * 0.6 > g and b * 0.6 > r:
                continue
            else:
                grayValue = int(0.299 * r + 0.587 * g + 0.114 * b)
                result[i, j] = [grayValue, grayValue, grayValue]

    return result

def contrastBrightness(imagePath, contrast=100, brightness=40):
    img = cv2.imread(imagePath) 
    result = np.array(img, dtype=np.int32)
    height, width = img.shape[:2]

    for i in range(height):
        for j in range(width):
            b, g, r = int(img[i, j, 0]), int(img[i, j, 1]), int(img[i, j, 2])
            isBlue = b > 100 and b * 0.6 > g and b * 0.6 > r
            isYellow = (b + g) * 0.3 > r
            if isBlue or isYellow:
                for channel in range(3):
                    oldValue = img[i, j, channel]
                    newValue = (oldValue - 127) * (contrast/127 + 1) + 127 + brightness
                    result[i, j, channel] = newValue

    result = np.clip(result, 0, 255)
    result = np.array(result, dtype=np.uint8)
    return result

def bilinearInterpolation(imagePath, scaleFactor):
    img = cv2.imread(imagePath)
    originalHeight, originalWidth = img.shape[:2]
    newHeight = int(originalHeight * scaleFactor)
    newWidth = int(originalWidth * scaleFactor)
    channels = img.shape[2]
    result = np.zeros((newHeight, newWidth, channels), dtype=np.uint8)

    for i in range(newHeight):

        for j in range(newWidth):
            x = j / scaleFactor
            y = i / scaleFactor
            x1 = int(math.floor(x))
            x2 = min(x1 + 1, originalWidth - 1)
            y1 = int(math.floor(y))
            y2 = min(y1 + 1, originalHeight - 1)
            wx = x - x1
            wy = y - y1

            for c in range(channels):
                I11 = img[y1, x1, c]
                I12 = img[y1, x2, c]
                I21 = img[y2, x1, c]
                I22 = img[y2, x2, c]
                value = (1 - wx) * (1 - wy) * I11 + wx * (1 - wy) * I12 + \
                        (1 - wx) * wy * I21 + wx * wy * I22
                result[i, j, c] = int(value)

    return result

def edgeDetection(imagePath):
    img = cv2.imread(imagePath) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    sobelx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float32)

    sobely = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)

    blurredFloat = blurred.astype(np.float32)
    gradientx = cv2.filter2D(blurredFloat, -1, sobelx)
    gradienty = cv2.filter2D(blurredFloat, -1, sobely)
    gradientMagnitude = np.sqrt(gradientx.astype(np.float32)**2 + gradienty.astype(np.float32)**2)
    gradientMagnitude = np.clip(gradientMagnitude, 0, 255)
    gradientMagnitude = gradientMagnitude.astype(np.uint8)

    return gradientMagnitude

def main():
    q1 = "q1.jpg"  
    q2 = "q2.jpg"  
    q3 = "q3.jpg"  

    result1_1 = grayscaleColorFilter(q1)
    cv2.imwrite("result1-1.jpg", result1_1)
    cv2.imshow("result1-1", result1_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result1_2 = contrastBrightness(q1, contrast=100, brightness=40)
    cv2.imwrite("result1-2.jpg", result1_2)
    cv2.imshow("result1-2", result1_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result2 = bilinearInterpolation(q2, scaleFactor=3)
    cv2.imwrite("result2.jpg", result2)
    cv2.imshow("result2", result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    result3 = edgeDetection(q3)
    cv2.imwrite("result3.jpg", result3)
    cv2.imshow("result3", result3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()