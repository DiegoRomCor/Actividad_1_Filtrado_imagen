import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract

img = cv2.imread('placa_1.jpg')

inverted = cv2.bitwise_not(img)

hsv_inverted = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

min_blanco = np.array([0, 0, 200])
max_blanco = np.array([180, 20, 255])
mask = cv2.inRange(hsv_inverted, min_blanco, max_blanco)

result = cv2.bitwise_and(inverted, inverted, mask=mask)

result = cv2.GaussianBlur(result, (5,5), 0)

kernel = np.ones((10, 10), np.uint8)

img_erosion = cv2.erode(result, kernel, iterations=1)
img_dilation = cv2.dilate(result, kernel, iterations=1)

# Mostrar resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title('RGB')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Filtrado')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(img_erosion, cmap='gray')
plt.title("After Erosion")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(img_dilation, cmap='gray')
plt.title("After Dilation")
plt.axis('off')

plt.show()

print("---ORIGINAL---")

custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
text = pytesseract.image_to_string(img, config = custom_config)
text = ''.join([c for c in text if c.isalnum()])
print(text)

print("---EROSION+GAUSBLUR---")

text = pytesseract.image_to_string(img_erosion, config = custom_config)
text = ''.join([c for c in text if c.isalnum()])
print(text)

print("---DILATACION+GAUSBLUR---")

text = pytesseract.image_to_string(img_dilation, config = custom_config)
text = ''.join([c for c in text if c.isalnum()])
print(text)




img = cv2.imread('placa_4.jpg')

inverted = cv2.bitwise_not(img)

hsv_inverted = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

min_blanco = np.array([0, 0, 200])
max_blanco = np.array([180, 40, 255])
mask = cv2.inRange(hsv_inverted, min_blanco, max_blanco)

result = cv2.bitwise_and(inverted, inverted, mask=mask)

result = cv2.GaussianBlur(result, (5,5), 0)

kernel = np.ones((10, 10), np.uint8)

img_erosion = cv2.erode(result, kernel, iterations=1)
img_dilation = cv2.dilate(result, kernel, iterations=1)

# Mostrar resultados
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title('RGB')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Filtrado')
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(img_erosion, cmap='gray')
plt.title("After Erosion")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(img_dilation, cmap='gray')
plt.title("After Dilation")
plt.axis('off')

plt.show()

print("---ORIGINAL---")

custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
text = pytesseract.image_to_string(img, config = custom_config)
text = ''.join([c for c in text if c.isalnum()])
print(text)

print("---EROSION+GAUSBLUR---")

text = pytesseract.image_to_string(img_erosion, config = custom_config)
text = ''.join([c for c in text if c.isalnum()])
print(text)

print("---DILATACION+GAUSBLUR---")

text = pytesseract.image_to_string(img_dilation, config = custom_config)
text = ''.join([c for c in text if c.isalnum()])
print(text)