# Laplacian Sharpening from Scratch

A pure NumPy implementation of Laplacian sharpening for image enhancement, featuring manual convolution and replicate padding without using OpenCV's built-in functions.

## Features
- **Manual 8-neighborhood Laplacian kernel** implementation
- **Replicate padding** implemented from scratch
- **Pure NumPy convolution** without OpenCV's `filter2D`
- **Image sharpening** via edge enhancement
- **Visual comparison** of original, Laplacian, and sharpened images

## Algorithm
The implementation uses the standard 8-directional Laplacian kernel:
