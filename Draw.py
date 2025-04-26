import argparse
import json

import cv2
from pdf2image import convert_from_path

class Draw:
    def __init__(self, pdf_path, page_number=1, dpi=300):
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.pages = convert_from_path(pdf_path, dpi=dpi)

        # Make sure page_number is within bounds
        if page_number < 1 or page_number > len(self.pages):
            raise ValueError(f"Page number {page_number} is out of range. The document has {len(self.pages)} pages.")

        self.current_page = self.pages[page_number - 1]  # page_number is 1-indexed

        self.temp_image_path = "temp_page.png"
        self.selected_boxes = []

    def save_temp_image(self):
        """Save the current page as a temporary image."""
        self.current_page.save(self.temp_image_path)


    def interactive_zone_selection(self):
        """Allow the user to interactively select zones using OpenCV."""
        self.save_temp_image()
        image = cv2.imread(self.temp_image_path)

        screen_width = 1200  # Increased width for display
        screen_height = 900  # Increased height for display
        h, w, _ = image.shape
        scale = min(screen_width / w, screen_height / h)

        resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))

        def draw_rectangle(event, x, y, flags, param):
            nonlocal rect_start, rect_end, cropping, image_copy
            if event == cv2.EVENT_LBUTTONDOWN:
                rect_start = (int(x / scale), int(y / scale))
                cropping = True
            elif event == cv2.EVENT_MOUSEMOVE and cropping:
                image_copy = resized_image.copy()
                rect_end = (int(x / scale), int(y / scale))
                cv2.rectangle(image_copy, (int(rect_start[0] * scale), int(rect_start[1] * scale)),
                              (int(rect_end[0] * scale), int(rect_end[1] * scale)), (0, 255, 0), )
                cv2.imshow("Select Zone", image_copy)
            elif event == cv2.EVENT_LBUTTONUP:
                rect_end = (int(x / scale), int(y / scale))
                cropping = False
                box = (rect_start[0], rect_start[1], rect_end[0], rect_end[1])
                self.selected_boxes.append(box)
                formatted_box = {
                    "x1": box[0],
                    "x2": box[2],
                    "y1": box[1],
                    "y2": box[3]
                }
                print(json.dumps(formatted_box))
                # {"x1": 405, "x2": 2415, "y1": 342, "y2": 405}
                cv2.rectangle(resized_image, (int(rect_start[0] * scale), int(rect_start[1] * scale)),
                              (int(rect_end[0] * scale), int(rect_end[1] * scale)), (0, 255, 0), 1)
                cv2.imshow("Select Zone", resized_image)

        rect_start, rect_end = (0, 0), (0, 0)
        cropping = False
        image_copy = resized_image.copy()

        cv2.imshow("Select Zone", resized_image)
        cv2.setMouseCallback("Select Zone", draw_rectangle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Extract text from a PDF using interactive zone selection.")
    parser.add_argument("pdf_file", help="Path to the input PDF file")
    parser.add_argument("--page", type=int, default=1, help="Page number to select zones from (default: 1)")
    args = parser.parse_args()

    pdf_extractor = Draw(args.pdf_file, page_number=args.page)
    pdf_extractor.interactive_zone_selection()

if __name__ == "__main__":
    main()