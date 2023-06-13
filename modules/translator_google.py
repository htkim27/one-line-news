from googletrans import Translator

class GoogleTranslator:
    def __init__(self) -> None:
        self.translator = Translator()
        
    def translate(self, 
                  src_text:str = "",
                  src:str = "ko",
                  dest:str = "en"):
        
        translated_text = self.translator.translate(src_text, src=src, dest=dest).text
        
        return translated_text
    
if __name__ == "__main__":
    
    text = "나는 배가 고프다"
    translator = GoogleTranslator()
    res = translator.translate(text)
    print(res)