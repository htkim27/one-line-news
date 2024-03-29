from transformers import pipeline, AutoModelForCausalLM
import torch

PROMPT_TEMPLATE = "### 명령문: 키워드 <{keys_len}>개를 기반으로 뉴스 기사 제목을 만들어줘.\n\n\n### 키워드: {keywords}\n\n\n### 답변: "

class OneLineNewsGenerator:
    
    def __init__(self, model_path:str, load_in_8bit:bool=False)->None:
        
        # one line text news generator
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype = torch.float16,
            # revision="fp16",
            low_cpu_mem_usage = True,
            load_in_8bit=load_in_8bit,
            # revision="8bit",
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=model_path,
        )
        
    def generate(
        self,      
        keywords:str,
        do_sample:bool=True,
        max_new_tokens:int=32,
        temperature:float=0.4,
        top_p:float=0.95,
        return_full_text:bool=False,
        eos_token_id:int=2,
        no_repeat_ngram_size:int=2)->str:
        
        keys_len = len(keywords.split(","))
        
        prompt = PROMPT_TEMPLATE.format(keys_len = keys_len, keywords = keywords)
        
        ans = self.pipe(
            prompt,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_full_text=return_full_text,
            eos_token_id=eos_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        
        msg = ans[0]["generated_text"]
        
        # 프롬프트 영향으로 결과에 ### 등장시 split
        if "###" in msg:
            one_line_news = msg.split("###")[0]
        else:
            one_line_news = msg
            
        return one_line_news