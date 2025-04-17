import cv2
from PIL import Image
import numpy as np
import re
from paddleocr import PaddleOCR

CHAR_MAP = {
    '+': '7',
    'O': '0', 'o': '0', '.': '0',
    'I': '1', 'l': '1', 'X': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8',
    'A': '4', 'a': '0',
    'G': '4',
    'T': '1',
    'Q': '9', 'q': '4',
    'U': '1', 'u': '4',
    'c': '0',
    'F': '7',
}

class DateOCRProcessor:
    def __init__(
            self,
            upscale_factor: float = 2.0,
            denoise_h: int = 9,
            clahe_clip: float = 2.1,
            clahe_grid: tuple = (8, 8),
            threshold_block: int = 41,
            threshold_C: int = 9
    ):

        # Configuration parameters
        self.image_path = None
        self.image_array = None  # Ajout pour l'image directe
        self.det_model_dir = "./inference/en_PP-OCRv3_det_slim_infer/"
        self.cls_model_dir = "./inference/ch_ppocr_mobile_v2.0_cls_infer/"
        self.rec_model_dir = "./inference/en_PP-OCRv3_rec_infer/"
        self.rec_char_dict_path = "./en_dict.txt"
        self.use_gpu = False

        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=True,
            use_gpu=self.use_gpu,
            det_model_dir=self.det_model_dir,
            rec_model_dir=self.rec_model_dir,
            cls_model_dir=self.cls_model_dir,
            rec_char_dict_path=self.rec_char_dict_path,
        )

        # Preprocessing parameters
        self.upscale_factor = upscale_factor
        self.denoise_h = denoise_h
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.threshold_block = threshold_block
        self.threshold_C = threshold_C
        self.image_array = None
        self.image_path = None

        self.date_pattern = re.compile(r'\b(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/\d{4}\b')

    def preprocess_image(self) -> Image.Image:
        """
        Enhance image for OCR:
          1. Load image from path
          2. Upscale for resolution
          3. Convert to grayscale
          4. Denoise (edge-preserving)
          5. CLAHE for contrast
          6. Unsharp masking
          7. Adaptive threshold
          8. Morphological closing
        Returns a PIL Image (binary) for OCR.
        """
        if self.image_array is not None:
            img = self.image_array
        elif self.image_path is not None:
            print(self.image_path)
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        else:
            raise ValueError("No image provided for OCR")

        if img is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1)
        img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

        # Upscale
        if self.upscale_factor != 1.0:
            img = cv2.resize(
                img, None,
                fx=self.upscale_factor,
                fy=self.upscale_factor,
                interpolation=cv2.INTER_CUBIC
            )

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Denoise
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=self.denoise_h, sigmaSpace=75)
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        gray = clahe.apply(gray)
        # Unsharp mask
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.threshold_block,
            self.threshold_C
        )
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite("processed_image.png", binary)  # Save for debugging
        return Image.fromarray(binary)

    def correct_date_text(self, text, two_digit_year_cutoff=30):
        """
        Clean up OCR'd text and return a date in dd/mm/YYYY or the cleaned string if no valid date.
        two_digit_year_cutoff: max two-digit year to map to 20xx (others map to 19xx).
        """
        corrected = ''.join(CHAR_MAP.get(c, c) for c in text)
        corrected_list = list(corrected)

        # 2) Prepare raw digits-only string
        # withoutslash = ''.join(corrected_list) \
        #     .replace('/', '').replace('-', '') \
        #     .replace(' ', '').replace(':', '') \
        #     .replace('.', '').replace(',', '') \
        #     .replace(';', '').replace("'", '').replace('"', '').replace('+', '')
        # import re

        withoutslash = re.sub(r'\D', '', ''.join(corrected_list))

        for pos in ([1, 4] if len(withoutslash) == 7 else [2, 5] if len(withoutslash) >= 8 else []):
            if pos < len(corrected_list) and corrected_list[pos] in {'1', '7', 'f'}:
                corrected_list[pos] = '/'

        # visually = ''.join(corrected_list)

        print("Digits only:", withoutslash)
        # if contains caracters withoutslash and not only digits return corrected
        if any(c not in '0123456789' for c in withoutslash) or len(text) > 15:
            return text  # too many non-digits

        n = len(withoutslash)

        if n == 6:
            # ddmmyy
            day, month, year = withoutslash[:2], withoutslash[2:4], withoutslash[4:6]
        elif n == 7:
            # assume one extra rogue digit—drop it and grab dd, mm, yy
            day, month, year = withoutslash[:2], withoutslash[2:4], withoutslash[-2:]
        elif n >= 8:
            s = withoutslash
            day = s[:2]
            year = s[-4:]
            mid = s[2:-4]  # everything between day and year

            # look for all two‑digit runs in `mid` that form a valid month 01–12
            candidates = [
                mid[i:i + 2]
                for i in range(len(mid) - 1)
                if 1 <= int(mid[i:i + 2]) <= 12
            ]

            if candidates:
                # if there’s more than one, pick the numerically smallest (so "04" beats "10")
                month = min(candidates, key=lambda mm: int(mm))
            else:
                # fallback if OCR was really messy
                month = s[2:4]
        else:
            return corrected  # too few digits

        if len(year) == 2:
            yy = int(year)
            century = '20' if yy <= two_digit_year_cutoff else '19'
            year = century + year

        from datetime import datetime
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime('%d/%m/%Y')
        except Exception:
            return corrected  # invalid date

    def find_date(self):
        processed_image = self.preprocess_image()

        result = self.ocr.ocr(np.array(processed_image), cls=False)

        print("OCR Result: ", result)

        # Handle no-detection case
        if not result or result[0] is None:
            return "Aucune date trouvée ou date non reconnue", None

        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            print("Detected text: ", text, "Confidence: ", confidence)
            match = self.date_pattern.search(text)
            if match:
                return match.group(), confidence

            corrected = self.correct_date_text(text)

            print("Corrected text: ", corrected)

            if corrected is None:
                continue

            match = self.date_pattern.search(corrected)
            if match:
                return match.group(), confidence

        return "Aucune date trouvée ou date non reconnue", None

    def run(self, image_input):
        """
        Run OCR on either an image path (str) or a NumPy image (array).
        """
        if isinstance(image_input, str):
            self.image_path = image_input
            self.image_array = None
        elif isinstance(image_input, np.ndarray):
            self.image_array = image_input
            self.image_path = None
        else:
            raise ValueError("image_input must be a file path or a NumPy array")

        return self.find_date()