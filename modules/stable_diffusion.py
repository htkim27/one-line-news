import torch
from diffusers import StableDiffusionPipeline

class  StableDiffusion:
    def __init__(self, model_id):
        device="cuda"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                                            revision='fp16',
                                                            torch_dtype=torch.float16,
                                                            safety_checker=None,)
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
        self.pipe = pipe

       
    def generate(self, prompt):
        image = self.pipe(prompt).images[0]  
        return image
        
if __name__ == "__main__":
    generator = StableDiffusion(model_id ="CompVis/stable-diffusion-v1-4")
    output = generator.generate('여자')