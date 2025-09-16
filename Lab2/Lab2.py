import cv2
import numpy as np

def calc_hist(image, channel=0):
    h = np.zeros(256, dtype=int)
    for value in image[..., channel].flatten():
        h[value] += 1
    return h

def calc_cdf(hist):
    cdf = np.cumsum(hist)
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    return cdf_normalized.astype(np.uint8)

img = cv2.imread('histogram.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 1-1
bgr_eq = img.copy()
for c in range(3):
    hist = calc_hist(img, c)
    cdf = calc_cdf(hist)
    bgr_eq[..., c] = cdf[img[..., c]]
cv2.imwrite('result1-1.jpg', bgr_eq)

# 1-2
v = hsv[..., 2]
hist_v = calc_hist(hsv, 2)
cdf_v = calc_cdf(hist_v)
hsv[..., 2] = cdf_v[v]
hsv_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('result1-2.jpg', hsv_eq)

# 2
gray = cv2.imread('otsu.jpg', cv2.IMREAD_GRAYSCALE)
total_pixels = gray.size
min_within_var = float('inf')
best_t = 0

for t in range(256):
    group1 = gray[gray <= t]
    group2 = gray[gray > t]
    if len(group1) == 0 or len(group2) == 0:
        continue
    var1 = np.var(group1)
    var2 = np.var(group2)
    within_var = var1 + var2
    if within_var < min_within_var:
        min_within_var = within_var
        best_t = t

otsu_result = np.zeros_like(gray)
otsu_result[gray > best_t] = 255
cv2.imwrite('result2.jpg', otsu_result)
