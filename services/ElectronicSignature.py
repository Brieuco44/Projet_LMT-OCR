import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

class ElectronicSignature:

    def __init__(self, ratio):
        self.ratio = ratio

    def is_signature_present(self, image_input):
        try:
            # Check if input is a string (assume file path) or ndarray (image)
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    print("Error: Could not read the image.")
                    return (False, 0.0) if self.ratio else False
            elif isinstance(image_input, np.ndarray):
                image = image_input
            else:
                raise ValueError("Input must be a file path or an OpenCV image (numpy array).")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Edge detection to find the signature box
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If no contours found, return False
            if not contours:
                print("No contours detected.")
                if self.ratio:
                    return False, 0.0
                else:
                    return False

            # Find the largest rectangle contour
            roi_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(roi_contour)

            # Crop the image to the detected ROI
            roi_image = gray[y:y + h, x:x + w]

            # Thresholding to highlight the signature
            _, thresh = cv2.threshold(roi_image, 150, 255, cv2.THRESH_BINARY_INV)

            # Morphological transformation to enhance the signature
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Count non-zero pixels (indicative of ink presence)
            ink_pixels = cv2.countNonZero(morphed)
            signature_ratio = ink_pixels / (w * h)

            # Define a higher threshold ratio to reduce sensitivity
            signature_threshold = 0.05  # Increased from 0.02 to 0.05

            # Check if the ratio of ink to area is significant
            is_present = signature_ratio > signature_threshold
            if self.ratio:
                return is_present, signature_ratio
            else:
                return is_present

        except Exception as e:
            print(f"Error during processing: {e}")
            if self.ratio:
                return False, 0.0
            else:
                return False



    def test_ispresent(self):
        # Expected results for the images

        tests = [
            {"image": "../tests/img/sign1.png", "expected": True},
            {"image": "../tests/img/sign2.png", "expected": True},
            {"image": "../tests/img/sign3.png", "expected": False},
            {"image": "../tests/img/sign4.png", "expected": False},
            {"image": "../tests/img/sign5.png", "expected": False},
            {"image": "../tests/img/sign6.png", "expected": False},
            {"image": "../tests/img/sign7.png", "expected": True}, # Vrai car signature manuscrite
            {"image": "../tests/img/sign8.png", "expected": True} # Vrai car signature manuscrite

        ]

        # Rich Console setup
        console = Console()
        table = Table(title="Signature Detection Results")
        table.add_column("Image", justify="left", style="cyan")
        table.add_column("Expected", justify="center", style="green")
        table.add_column("Got", justify="center", style="red")
        if self.ratio:
            table.add_column("Signature Ratio", justify="center", style="magenta")
        table.add_column("Status", justify="center", style="bold yellow")

        # Execute detection and populate the table
        for test in tests:
            if self.ratio:
                result, ratio = self.is_signature_present(test["image"])
                status = "✅" if result == test["expected"] else "❌"
                table.add_row(test["image"], str(test["expected"]), str(result), f"{ratio:.4f}", status)
            else:
                result = self.is_signature_present(test["image"])
                status = "✅" if result == test["expected"] else "❌"
                table.add_row(test["image"], str(test["expected"]),str(result),status)

        # Display the table
        console.print(table)
