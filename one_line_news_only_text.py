import re
from time import time

import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM
# import kss

MODEL = "./checkpoints/checkpoint-230"
# load gpt model
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype = torch.float16,
    revision="fp16",
    low_cpu_mem_usage = True,
    # load_in_8bit=True,
    # revision="8bit",
)

# load text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=MODEL,
)

# chatbot function
def answer(state, state_chatbot, text):    
    start = time()

    print("context : ", state)
    
    query = text

    prompt = f"### 명령문: 키워드를 기반으로 뉴스 기사 제목을 만들어줘.\n\n\n### 키워드: {query}\n\n\n### 답변: "
    print(prompt)
    
    # run gpt model
    ans = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=32,
        temperature=0.4,
        top_p=0.95,
        return_full_text=False,
        eos_token_id=2,
        no_repeat_ngram_size=2,
    )
    msg = ans[0]["generated_text"]
    
    # 프롬프트 영향으로 결과에 ### 등장시 split
    if "###" in msg:
        msg = msg.split("###")[0]

    # calculate elapsed time
    end = time()
    elapsed_time = f"\n\n \n\n \n\n<생성에 걸린 시간 : {round(end-start)} 초>"
    
    # state_chatbot : displayed
    state_chatbot = [(text, msg+"\n\n"+elapsed_time)]
    
    # state : 저장되는 맥락
    # 4개 기억 -> 가장 오래 된 기억 삭제

    state = []

    return state, state_chatbot, state_chatbot

############## Gradio Things ##############

with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State([])
    state_chatbot = gr.State([])

    with gr.Row():
        gr.HTML(
            """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h1>One Line News</h1>
                <h2>Economy News Title Generation With Keywords</h2>

            </div>
            <div>
                Built by Heywon & htkim
            </div>
        </div>"""
        )

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")

    # with gr.Row():
    #     cont = gr.Radio(["ON", "OFF"], label="이전 맥락 반영", info="off를 누르면 완전 reset입니다!\n24문장까지 반영")

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="키워드는 콤마(,)로 구분해주세요").style(
            container=False
        )

    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot], scroll_to_output=True)
    txt.submit(lambda: "", None, txt, scroll_to_output=True)

# share=True : 외부 IP에서 접근 가능한 public link 생성
demo.launch(debug=True, 
            server_name="0.0.0.0", 
            server_port=8882,
            share=True
            )
