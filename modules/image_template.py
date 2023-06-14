from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

class ImageTemplate:
    
    def __init__(self,
                 width_margin:int=50,
                 height_margin:int=130,
                 image_width_position:float=1/2,
                 image_height_postion:float=1/4,
                 font_size:int = 24,
                 font_color:Tuple[int] = (0, 0, 0),
                 font_path:str = ""
                 
                 ) -> None:
        
        self.width_margin = width_margin
        self.height_margin = height_margin
        self.image_width_positon = image_width_position
        self.image_height_position = image_height_postion
        
        self.font_size = font_size
        self.font_color = font_color
        self.font_path = font_path
    
    def make(self,
             image:Image.Image,
             text:str) -> Image.Image:
        
        new_width = image.width + self.width_margin
        new_height = image.height + self.height_margin
        
        # image
        canvas = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        
        x = int((new_width - image.width) * self.image_width_positon)
        y = int((new_height - image.height) * self.image_height_position)
        
        canvas.paste(image, (x, y))

        # text
        text_position = (x, int(new_height - (new_height-image.height)/2))

        font = ImageFont.truetype(self.font_path, self.font_size, encoding="unic")  # Replace with the actual font file path
        draw = ImageDraw.Draw(canvas)
        draw.text(text_position, text, font=font, fill=self.font_color)
        
        return canvas
    
if __name__ == "__main__":
    
    image = Image.open("./test3.png")
    text = "주식 심각한 하락, 밥을 먹다가 깜짝 놀라"
    
    
    image_template = ImageTemplate(font_path="./korean_font.ttf")
    image = image_template.make_image_template(image, text)
    
    image.save("./test3_out.png")
        