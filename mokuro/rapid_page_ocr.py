import math

from loguru import logger
from rapidocr import RapidOCR
from rapidocr.main import RapidOCROutput

from mokuro.utils import imread


class RapidPageOcr:
    def __init__(self):
        self.ocr = RapidOCR()
        logger.info("RapidOCR model initialized")

    def __call__(self, img_path):
        img = imread(img_path)
        if img is None:
            raise Exception()
        H, W, *_ = img.shape
        result = {"version": "0.2.2", "img_width": W, "img_height": H, "blocks": []}

        ocr_output = self.ocr(img_path)
        if isinstance(ocr_output, RapidOCROutput):
            txts = ocr_output.txts
            boxes = ocr_output.boxes
            if txts is not None and boxes is not None:
                for _, (box, txt) in enumerate(zip(boxes, txts)):
                    box_width = self.get_box_width(box)
                    box_height = self.get_box_height(box)
                    if box_height > 1.1 * box_width:
                        vertical = True
                        font_size = max(int(box_width * 0.9), 10)
                    else:
                        vertical = False
                        font_size = max(int(box_height * 0.8), 10)
                    result_blk = {
                        "box": [
                            self.get_box_min_x(box),
                            self.get_box_min_y(box),
                            self.get_box_max_x(box),
                            self.get_box_max_y(box),
                        ],
                        "vertical": vertical,
                        "font_size": font_size,
                        "lines_coords": [box],
                        "lines": [txt],
                    }
                    result["blocks"].append(result_blk)

        return result

    @staticmethod
    def get_box_height(box) -> float:
        return math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)

    @staticmethod
    def get_box_width(box) -> float:
        return math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)

    @staticmethod
    def get_box_min_x(box) -> float:
        return min(point[0] for point in box)

    @staticmethod
    def get_box_max_x(box) -> float:
        return max(point[0] for point in box)

    @staticmethod
    def get_box_min_y(box) -> float:
        return min(point[1] for point in box)

    @staticmethod
    def get_box_max_y(box) -> float:
        return max(point[1] for point in box)
