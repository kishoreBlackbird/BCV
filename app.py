import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Image Filters App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load_image(uploaded_file)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        filter_type = st.selectbox("Select Filter Type:", ["Edge Detection", "Line Detection", "Corner Detection"])

        if filter_type == "Edge Detection":
            edge_filter = st.selectbox("Select Edge Filter:", ["Canny", "LOG", "DOG"])
        elif filter_type == "Line Detection":
            edge_filter = "Hough Transform"
        elif filter_type == "Corner Detection":
            edge_filter = st.selectbox("Select Corner Filter:", ["Harris", "Hessian Affine"])

        if st.button("Apply Filter"):
            result_image = apply_filter(image, filter_type.lower(), edge_filter.lower())
            st.image(result_image, caption="Result Image", use_column_width=True)

def load_image(uploaded_file):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), 1)
    return image

def apply_filter(image, filter_type, filter_method):
    if filter_type == "edge detection":
        return apply_edge_detection(image, filter_method)
    elif filter_type == "line detection":
        return apply_line_detection(image, filter_method)
    elif filter_type == "corner detection":
        return apply_corner_detection(image, filter_method)
    else:
        return image

def apply_edge_detection(image, filter_method):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if filter_method == "canny":
        return cv2.Canny(gray, 50, 150)
    elif filter_method == "log":
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    elif filter_method == "dog":
        return cv2.filter2D(gray, -1, create_dog_kernel())

def create_dog_kernel():
    kernel_size = 5
    sigma1 = 1.0
    sigma2 = 2.0

    kernel1 = cv2.getGaussianKernel(kernel_size, sigma1)
    kernel2 = cv2.getGaussianKernel(kernel_size, sigma2)

    dog_kernel = kernel1 - kernel2.T
    return dog_kernel

def apply_line_detection(image, filter_method):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    if filter_method == "hough transform":
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        result_image = image.copy()

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return result_image

def apply_corner_detection(image, filter_method):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if filter_method == "harris":
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        result_image = image.copy()
        result_image[dst > 0.01 * dst.max()] = [0, 0, 255]
        return result_image
    elif filter_method == "hessian affine":
        hessian = cv2.ximgproc.createHessianAffineDetector()
        keypoints = hessian.detect(gray)
        result_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return result_image
    else:
        return image

if __name__ == "__main__":
    main()