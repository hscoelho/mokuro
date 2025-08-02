import os

from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image

from mokuro import __version__


class GooglePageVision:
    def __init__(self):
        try:
            # authenticates
            print("Parsing Google credentials")
            # TODO: Add api key option
            # service account file
            google_credentials_file = os.path.join(os.path.expanduser("~"), ".config", "google_vision.json")

            google_credentials = service_account.Credentials.from_service_account_file(google_credentials_file)
            self.client = vision.ImageAnnotatorClient(credentials=google_credentials)
            print("Google vision ready")
        except Exception as e:
            print("Failed to initialize google vision client")
            print(e)

    def __call__(self, img_path):
        with open(img_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        try:
            # There is another Google Cloud Vision text detection function called "document_text_detection()"
            # No difference was found between them for manga, but more tests are needed
            response = self.client.text_detection(image=image)
        except Exception as e:
            print(e)
            return
        txt_blocks = self.parse_api_response(response)
        img = Image.open(img_path)
        img_width, img_height = img.size
        result = {
            "version": __version__,
            "img_width": img_width,
            "img_height": img_height,
            "blocks": txt_blocks,
        }
        return result

    def parse_api_response(self, response):
        text_blocks = []
        document = response.full_text_annotation

        for page in document.pages:
            for block in page.blocks:
                lines = []
                txt = ""
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            bound = symbol.bounding_box
                            txt += symbol.text
                            detected_break = symbol.property.detected_break
                            if detected_break is not None:
                                if detected_break.type_ == 5:
                                    lines.append(txt)
                                    txt = ""
                                if detected_break.type_ == 3:
                                    lines.append(txt)
                                    txt = ""
                if txt != "":
                    lines.append(txt)

                if len(lines) == 0:
                    continue

                bound = block.bounding_box
                lines_coords = [
                    [bound.vertices[0].x, bound.vertices[0].y],
                    [bound.vertices[1].x, bound.vertices[1].y],
                    [bound.vertices[2].x, bound.vertices[2].y],
                    [bound.vertices[3].x, bound.vertices[3].y],
                ]
                box = [
                    min(self.get_x_coords(lines_coords)),
                    min(self.get_y_coords(lines_coords)),
                    max(self.get_x_coords(lines_coords)),
                    max(self.get_y_coords(lines_coords)),
                ]
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]

                assumed_vertical_line_width = box_width / len(lines)
                if box_height * 1.1 > assumed_vertical_line_width:
                    vertical = True
                    font_size = max(int((box_width / len(lines)) * 0.9), 10)
                else:
                    vertical = False
                    font_size = max(int((box_height / len(lines)) * 0.9), 10)

                text_block = {
                    "box": box,
                    "vertical": vertical,
                    "font_size": font_size,
                    "lines_coords": [lines_coords],
                    "lines": lines,
                }
                text_blocks.append(text_block)
        return text_blocks

    @staticmethod
    def get_x_coords(coords):
        return [point[0] for point in coords]

    @staticmethod
    def get_y_coords(coords):
        return [point[1] for point in coords]
