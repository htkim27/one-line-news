import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler

from PIL import Image

NUM_INFERENCE_STEPS = 20
NEGATIVE_PROMPT = "unnatural, unrealistic, cartoon, illustration, painting, drawing, unreal engine, black and white, monochrome, oversaturated, low saturation, surreal, underexposed, overexposed, jpeg artifacts, conjoined, aberrations, multiple levels, harsh lighting, anime, sketches, twisted, video game, photoshop, creative, UI, abstract, collapsed, rotten, ugly, bad anatomy, extra windows, text, watermark, out of frame"

class StableDiffusion:
    def __init__(self, model_id:str) -> None:
        device="cuda"
        
        if ".safetensors" in model_id:
            use_safetensors = True
        else:
            use_safetensors = False
            
        print(use_safetensors)
        
        pipe = StableDiffusionPipeline.from_pretrained(model_id, 
                                                            revision='fp16',
                                                            torch_dtype=torch.float16,
                                                            safety_checker=None,
                                                            use_safetensors = use_safetensors)
                
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention() 
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config
            )
        self.pipe = pipe

    def generate(self, prompt:str) -> Image.Image:
        image = self.pipe(prompt=prompt, 
                          negative_prompt=NEGATIVE_PROMPT,
                          num_inference_steps=NUM_INFERENCE_STEPS).images[0]  
        return image
        
if __name__ == "__main__":
    generator = StableDiffusion(model_id ="CompVis/stable-diffusion-v1-4")
    output = generator.generate('사람')