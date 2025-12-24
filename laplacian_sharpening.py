import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Read image in grayscale and normalize to [0, 1]
# --------------------------------------------------
img = cv2.imread("moon.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
h, w = img.shape

# --------------------------------------------------
# Laplacian mask (8-neighborhood)
# Formula equivalence:
# f(x,y) - sum(neighbors)  ==  8*f(x,y) - sum(neighbors)
# --------------------------------------------------
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# --------------------------------------------------
# Manual replicate padding implementation
# --------------------------------------------------
pad_img = np.zeros((h + 2, w + 2), dtype=np.float32)
pad_img[1:h+1, 1:w+1] = img

# Replicate first and last rows
pad_img[0, 1:w+1]   = img[0, :]
pad_img[h+1, 1:w+1] = img[-1, :]

# Replicate first and last columns
pad_img[1:h+1, 0]   = img[:, 0]
pad_img[1:h+1, w+1] = img[:, -1]

# Replicate corner pixels
pad_img[0, 0]       = img[0, 0]
pad_img[0, w+1]     = img[0, -1]
pad_img[h+1, 0]     = img[-1, 0]
pad_img[h+1, w+1]   = img[-1, -1]

# --------------------------------------------------
# Manual convolution (Laplacian)
# --------------------------------------------------
laplacian_img = np.zeros_like(img)

for i in range(h):
    for j in range(w):
        # 3x3 neighborhood around the pixel
        region = pad_img[i:i+3, j:j+3]

        # Center pixel f(x,y)
        center = img[i, j]

        # Sum of 8 neighbors
        neighbors_sum = (
            pad_img[i, j]     + pad_img[i, j+1] + pad_img[i, j+2] +   # Top row
            pad_img[i+1, j]   + pad_img[i+1, j+2] +                   # Middle row (excluding center)
            pad_img[i+2, j]   + pad_img[i+2, j+1] + pad_img[i+2, j+2] # Bottom row
        )

        # Laplacian computation:
        # f(x,y) - sum(neighbors)
        # Equivalent to: 8*f(x,y) - sum(neighbors)
        laplacian_img[i, j] = 8 * center - neighbors_sum

# --------------------------------------------------
# Sharpened image
# --------------------------------------------------
sharpened_img = img + laplacian_img

# --------------------------------------------------
# Clip values to valid range [0, 1]
# --------------------------------------------------
laplacian_img = np.clip(laplacian_img, 0, 1)
sharpened_img = np.clip(sharpened_img, 0, 1)

# --------------------------------------------------
# Display results
# --------------------------------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian_img, cmap='gray')
plt.title("Laplacian (8-directional)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharpened_img, cmap='gray')
plt.title("Sharpened Image")
plt.axis('off')

plt.tight_layout()
plt.show()
