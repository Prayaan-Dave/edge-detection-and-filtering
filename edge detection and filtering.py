import cv2
import numpy as np
import os

# ---------------------------
# Global variables
# ---------------------------
original_img = None
processed_img = None
edge_method = "Canny"
filter_method = "None"

# ---------------------------
# Processing Functions
# ---------------------------
def apply_edge_detection(img, method, params):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "Sobel":
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=params["ksize"])
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=params["ksize"])
        sobel = cv2.magnitude(dx, dy)
        return cv2.convertScaleAbs(sobel)
    elif method == "Laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=params["ksize"])
        return cv2.convertScaleAbs(lap)
    elif method == "Canny":
        return cv2.Canny(gray, params["threshold1"], params["threshold2"])
    else:
        return gray

def apply_filter(img, method, params):
    if method == "Gaussian":
        return cv2.GaussianBlur(img, (params["kernel"], params["kernel"]), 0)
    elif method == "Median":
        return cv2.medianBlur(img, params["kernel"])
    else:
        return img

# ---------------------------
# Trackbar Callbacks
# ---------------------------
def update_processing(_=None):
    global processed_img
    params = {
        "threshold1": cv2.getTrackbarPos("Canny Th1", "Processed"),
        "threshold2": cv2.getTrackbarPos("Canny Th2", "Processed"),
        "ksize": cv2.getTrackbarPos("Kernel Size", "Processed") | 1,  # ensure odd
        "kernel": cv2.getTrackbarPos("Filter Kernel", "Processed") | 1
    }
    img_filtered = apply_filter(original_img, filter_method, params)
    edges = apply_edge_detection(img_filtered, edge_method, params)
    processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    show_images()

def show_images():
    combined = np.hstack((original_img, processed_img))
    cv2.imshow("Processed", combined)

# ---------------------------
# Menu Functions
# ---------------------------
def choose_edge_method():
    global edge_method
    print("\nEdge Detection Methods:")
    print("1 - Sobel")
    print("2 - Canny")
    print("3 - Laplacian")
    choice = input("Choose method (1/2/3): ").strip()
    edge_method = {"1": "Sobel", "2": "Canny", "3": "Laplacian"}.get(choice, "Canny")

def choose_filter_method():
    global filter_method
    print("\nImage Filtering Methods:")
    print("0 - None")
    print("1 - Gaussian")
    print("2 - Median")
    choice = input("Choose filter (0/1/2): ").strip()
    filter_method = {"0": "None", "1": "Gaussian", "2": "Median"}.get(choice, "None")

# ---------------------------
# Main Program
# ---------------------------
def main():
    global original_img, processed_img

    # Load image
    path = input("Enter image path: ").strip()
    if not os.path.exists(path):
        print("Error: File not found.")
        return
    original_img = cv2.imread(path)
    if original_img is None:
        print("Error: Unable to load image.")
        return

    choose_filter_method()
    choose_edge_method()

    # Create window and trackbars
    cv2.namedWindow("Processed", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Canny Th1", "Processed", 50, 255, update_processing)
    cv2.createTrackbar("Canny Th2", "Processed", 150, 255, update_processing)
    cv2.createTrackbar("Kernel Size", "Processed", 3, 31, update_processing)
    cv2.createTrackbar("Filter Kernel", "Processed", 3, 31, update_processing)

    update_processing()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save processed image
            cv2.imwrite("processed_result.png", processed_img)
            print("Processed image saved as processed_result.png")
        elif key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
