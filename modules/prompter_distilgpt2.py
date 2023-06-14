from typing import Iterable

from transformers import GPT2Tokenizer, GPT2LMHeadModel

class DistilGPT2:
    """
   Anime style auto prompt engineering module
    """
    
    
    def __init__(self) -> None:
        """
        Init DistillGPT2 Model
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = GPT2LMHeadModel.from_pretrained('FredZhang7/distilgpt2-stable-diffusion')
        
    def generate(self, 
                 prompt:str = "",
                 temperature:float = 0.9,
                 top_k:int = 8,
                 max_length:int = 80,
                 repitition_penalty:float = 1.2,
                 num_return_sequences:int=1
                 ) -> Iterable[str]:
        """
        Args:
            prompt (str, optional): init prompt. Defaults to "".
            temperature (float, optional): a higher temperature will produce more diverse results, but with a higher risk of less coherent text. Defaults to 0.9.
            top_k (int, optional): the number of tokens to sample from at each step. Defaults to 8.
            max_length (int, optional): the maximum number of tokens for the output of the model. Defaults to 80.
            repitition_penalty (float, optional): the penalty value for each repetition of a token. Defaults to 1.2.
            num_return_sequences (int, optional): the number of results to generate. Defaults to 1.
        """   

        # generate the result with contrastive search
        input_ids = self.tokenizer(
            prompt, 
            return_tensors='pt').input_ids
        
        output = self.model.generate(
            input_ids, 
            do_sample=True, 
            temperature=temperature, 
            top_k=top_k, 
            max_length=max_length, 
            num_return_sequences=num_return_sequences, 
            repetition_penalty=repitition_penalty, 
            penalty_alpha=0.6,
            no_repeat_ngram_size=1, 
            early_stopping=True)
        
        for i in range(len(output)):        
            yield self.tokenizer.decode(output[i], skip_special_tokens=True)

if __name__ == "__main__":
    generator = DistilGPT2()
    results = generator.generate("apple, banana", num_return_sequences=1)
    
    [print(r) for r in results]