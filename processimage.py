import cv2
img = cv2.imread("thermal.jpg", cv2.IMREAD_GRAYSCALE)
img_color = cv2.applyColorMap(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_JET)
cv2.imwrite("C:/Users/Potatonet/Thermal_image/test_image/1.jpg", img_color)