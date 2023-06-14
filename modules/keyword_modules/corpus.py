import re
from typing import List
        
EMOJI_PATTERN = re.compile("["
                            u"\U00010000-\U0010FFFF"
                            "]+", flags=re.UNICODE)
        
def doc_loader(document:str)->List[str]:
    
    del_pattern = EMOJI_PATTERN
    
    processed_doc = []
    for line in document.split("\n"):
        processed_line = del_pattern.sub(r'', line)
        try:
            processed_doc.append(processed_line)
        except:
            continue
    
    return processed_doc 
    
        
    