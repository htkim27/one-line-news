from typing import List, Iterable

from time import sleep

import gradio as gr
from PIL import Image

from modules import (TextRankExtractor,
                     OneLineNewsGenerator, 
                     GoogleTranslator,
                     DistilGPT2, 
                     StableDiffusion,
                     ImageTemplate)


STOPWORDS_PATH = "./stopwordsKor.txt"
GPT_MODEL = "htkim27/one-line-news"
DISTILGPT_MODEL = "FredZhang7/distilgpt2-stable-diffusion-v2"
STABLE_DIFFUSION = "stabilityai/stable-diffusion-2-1-base"
TEXT_FONT = "./font/korean_font.ttf"


# init
keyword_extractor = TextRankExtractor(stopwords_path=STOPWORDS_PATH)
one_line_news_generator = OneLineNewsGenerator(model_path=GPT_MODEL)
translator = GoogleTranslator()
auto_prompter = DistilGPT2(model_id=DISTILGPT_MODEL)
stable_diffusion = StableDiffusion(model_id=STABLE_DIFFUSION)
image_template = ImageTemplate(font_path=TEXT_FONT)


# Keyword function
def keyword_func(document:str)->str:
    """

    Args:
        document (str): news or namu wiki document

    Returns:
        str: keywords delimitered with ,
    """
    keywords_l:List[str] = keyword_extractor.keyword_extract(document)
    keywords = ", ".join(keywords_l)
    
    return keywords

# One line news function
def one_line_news_func(keywords:str)->str:
    """

    Args:
        keywords (str): keywords from keyword_func

    Returns:
        str: one-line-news form keywords
    """
    one_line_news = one_line_news_generator.generate(keywords)
    
    return one_line_news

# Translator
def translator_func(keywords:str)->str:
    """

    Args:
        keywords (str): keywords from keyword_func

    Returns:
        str: keywords translated in English
    """
    translated_keywords = translator.translate(keywords)
    
    return translated_keywords

# Distil GPT
def auto_prompter_func(translated_keywords:str)->str:
    """

    Args:
        translated_keywords (str): keywords translated in English

    Returns:
        str: auto generated prompt from distilgpt2 - stable diffusion
    """
    prompts :List[str] = [prompt for prompt in auto_prompter.generate(translated_keywords)]
    prompt = prompts[0]
    
    return prompt

# Stable Diffusion
def stable_diffusion_func(prompt:str)->Image.Image:
    """

    Args:
        prompt (str): English prompt

    Returns:
        Image.Image: Generated One Image
    """
    
    image = stable_diffusion.generate(prompt)
    
    return image

# Image Template
def image_template_func(image:Image.Image,
                        one_line_news:str) -> Image.Image:
    """Make generated image into card-news shape

    Args:
        image (Image.Image): Generated Image
        one_line_news (str): Text

    Returns:
        Image.Image: Card-News Image
    """
    
    card_news = image_template.make(image=image, text=one_line_news)
    
    return card_news

def generate_card_news(state:gr.State, document:str):
    
    keywords = keyword_func(document)
    one_line_news = one_line_news_func(keywords)
    
    # 공짜 google translator 한 번 씩 에러가 남
    try:
        translated_keywords = translator_func(one_line_news+", "+keywords)
    except:
        print("Translator Error")
        sleep(3)
        translated_keywords = translator_func(one_line_news+", "+keywords)
        
    prompt = auto_prompter_func(translated_keywords)
    image = stable_diffusion_func(prompt)
    card_news = image_template_func(image, one_line_news)
    
    state = [card_news]
    
    return state, state

############## Gradio Things ##############

with gr.Blocks(css="#card_news .overflow-y-auto{height:2000px}") as demo:
    state = gr.State([])
    # state_chatbot = gr.State([])

    with gr.Row():
        gr.HTML(
            """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h1>One Line Card News</h1>
                <h2>Long Text News 2 Short Card News</h2>

            </div>
            <div>
                Built by yunhe1 & htkim27
            </div>
        </div>"""
        )

    with gr.Row():
        gallery = gr.Gallery(elem_id="card_news", 
                             label="Card-News", 
                             show_label=True)

    # with gr.Row():
    #     chatbot = gr.Chatbot(elem_id="keyword", label="Keywords")

    with gr.Row():
        document = gr.Textbox(label = "Input", 
                              show_label=True, 
                              placeholder="뉴스 기사 전체를 넣어주세요").style(
            container=False
        )
    image_path = "./deeptext_logo.png"
    link = "https://github.com/htkim27/one-line-news"
    with gr.Row():
        gr.HTML(
            f"""<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h2>DeepTextLab 2023</h2>
                <svg version="1.0" xmlns="http://www.w3.org/2000/svg"
                    width="300.000000pt" height="62.000000pt" viewBox="0 0 300.000000 62.000000"
                    preserveAspectRatio="xMidYMid meet">
                    <metadata>
                    Created by potrace 1.10, written by Peter Selinger 2001-2011
                    </metadata>
                    <g transform="translate(0.000000,62.000000) scale(0.100000,-0.100000)"
                    fill="#000000" stroke="none">
                    <path d="M393 597 c-10 -16 15 -33 29 -19 7 7 28 12 48 12 51 0 70 -24 70 -88
                    0 -29 5 -52 10 -52 6 0 10 24 10 53 0 71 -23 93 -103 99 -32 3 -60 1 -64 -5z"/>
                    <path d="M213 588 c-6 -7 -13 -29 -17 -48 -3 -19 -15 -43 -25 -55 -13 -14 -20
                    -41 -23 -92 -5 -65 -3 -73 12 -73 10 0 23 -7 30 -15 14 -17 60 -21 60 -5 0 6
                    -11 10 -24 10 -37 0 -50 23 -26 45 16 14 20 31 20 77 0 32 -4 58 -10 58 -6 0
                    -10 -26 -10 -59 0 -55 -16 -91 -33 -74 -15 14 -6 98 13 121 19 23 27 24 163
                    31 20 1 27 6 27 20 0 11 5 23 10 26 6 3 10 11 10 17 0 6 -9 3 -20 -7 -11 -10
                    -20 -24 -20 -32 0 -9 -15 -13 -54 -13 -75 0 -86 5 -77 38 7 26 11 27 87 33
                    l79 5 -81 2 c-55 1 -85 -2 -91 -10z"/>
                    <path d="M440 568 c0 -8 -11 -25 -25 -38 -21 -20 -25 -32 -25 -86 0 -72 -12
                    -87 -67 -82 l-38 3 -3 54 -3 54 45 -6 c36 -5 46 -4 46 8 0 18 -17 28 -25 15
                    -3 -5 -22 -10 -41 -10 l-34 0 0 -54 c0 -30 -5 -58 -10 -61 -8 -5 -7 -11 1 -21
                    6 -7 16 -11 22 -7 6 3 36 5 67 4 45 -2 59 2 74 18 11 12 16 29 13 41 -3 11 0
                    23 5 27 8 4 9 -10 4 -45 -6 -47 -5 -52 13 -52 11 0 23 -4 26 -10 3 -5 19 -10
                    36 -10 28 0 29 2 27 43 l-1 42 -4 -37 c-5 -47 -32 -51 -61 -10 -26 37 -25 195
                    1 200 14 3 17 -5 17 -41 0 -25 6 -49 13 -55 9 -8 11 2 9 46 -4 65 -6 68 -49
                    77 -25 5 -33 3 -33 -7z m10 -77 c0 -45 -2 -49 -21 -44 -19 5 -20 3 -14 -31 7
                    -36 -5 -66 -28 -66 -7 0 -7 4 1 12 7 7 12 39 12 72 0 48 4 64 22 83 12 13 23
                    23 25 23 2 0 3 -22 3 -49z"/>
                    <path d="M900 380 c0 -117 1 -120 25 -136 19 -12 36 -15 68 -10 54 9 84 29
                    101 69 13 28 14 29 15 9 1 -36 18 -59 51 -71 42 -14 84 -4 112 26 l23 26 -59
                    -8 c-44 -5 -61 -4 -67 7 -14 22 -11 23 54 24 60 1 63 0 77 -30 21 -44 62 -61
                    111 -47 21 7 44 21 50 32 10 19 8 20 -47 17 -46 -2 -59 0 -61 12 -3 12 10 15
                    62 17 l66 1 -2 -56 c-2 -32 0 -56 4 -54 4 1 19 2 33 2 23 0 26 3 22 25 -5 22
                    -3 25 16 19 36 -12 60 -6 84 21 56 65 -11 189 -80 146 -11 -7 -18 -7 -18 -1 0
                    5 -13 10 -30 10 -29 0 -30 -2 -31 -42 l0 -43 -16 32 c-17 33 -49 53 -87 53
                    -28 0 -76 -35 -76 -56 0 -8 -4 -14 -10 -14 -5 0 -10 6 -10 14 0 21 -48 56 -76
                    56 -38 0 -70 -20 -87 -53 l-16 -32 0 48 c-1 40 -6 54 -30 78 -27 27 -35 29
                    -100 29 l-71 0 0 -120z m131 64 c8 -10 14 -41 14 -75 0 -64 -13 -85 -55 -91
                    -25 -3 -25 -3 -28 90 l-3 92 29 0 c16 0 36 -7 43 -16z m197 -76 c3 -15 -4 -18
                    -32 -18 -38 0 -44 8 -24 28 17 17 52 11 56 -10z m180 10 c20 -20 14 -28 -23
                    -28 -24 0 -35 5 -35 14 0 24 39 33 58 14z m183 -12 c17 -37 4 -78 -24 -74 -17
                    2 -23 11 -25 36 -3 32 11 62 29 62 5 0 14 -11 20 -24z"/>
                    <path d="M1710 480 c0 -16 7 -20 35 -20 l35 0 0 -115 0 -115 30 0 30 0 0 115
                    0 115 40 0 c33 0 40 3 40 20 0 19 -7 20 -105 20 -98 0 -105 -1 -105 -20z"/>
                    <path d="M2466 379 c1 -74 6 -125 13 -132 6 -6 43 -13 81 -15 68 -4 70 -4 70
                    19 1 20 2 21 10 8 13 -22 52 -31 82 -20 14 5 47 4 82 -2 45 -8 61 -8 64 1 2 8
                    9 9 22 2 30 -16 69 5 86 47 17 40 13 83 -12 121 -13 20 -22 23 -55 20 l-39 -4
                    0 38 c0 36 -2 38 -32 38 l-33 0 3 -111 c2 -77 0 -109 -7 -104 -6 3 -11 33 -11
                    66 0 55 -2 60 -26 69 -40 16 -79 12 -108 -11 -14 -11 -26 -25 -26 -30 0 -13
                    47 -11 70 3 15 10 22 10 31 1 9 -9 5 -15 -17 -23 -62 -25 -79 -37 -82 -58 -3
                    -22 -3 -22 -98 -22 -4 0 -6 50 -5 110 l2 110 -33 0 -33 0 1 -121z m455 -15
                    c10 -27 5 -59 -14 -82 -25 -31 -50 43 -31 92 9 25 33 19 45 -10z m-181 -54 c0
                    -26 -38 -47 -46 -25 -7 18 12 45 32 45 8 0 14 -9 14 -20z"/>
                    <path d="M2280 460 c0 -29 -1 -30 -50 -30 -44 0 -51 -3 -64 -27 l-14 -28 -11
                    28 c-9 22 -16 27 -46 27 l-36 0 32 -46 c34 -50 34 -53 -11 -121 l-22 -33 34 0
                    c28 0 37 5 47 27 12 27 13 27 27 8 8 -10 14 -22 14 -27 0 -4 16 -8 35 -8 l36
                    0 -31 44 c-16 25 -30 49 -30 55 0 18 44 85 48 73 2 -7 12 -12 23 -12 17 0 19
                    -8 19 -64 0 -73 9 -85 70 -93 38 -5 40 -4 40 20 0 19 -6 26 -22 29 -20 3 -23
                    10 -26 56 -3 50 -2 52 22 52 19 0 26 5 26 20 0 15 -7 20 -25 20 -21 0 -25 5
                    -25 30 0 27 -3 30 -30 30 -27 0 -30 -3 -30 -30z"/>
                    <path d="M533 423 c-8 -20 5 -29 25 -18 9 6 32 9 50 7 26 -3 32 -7 32 -28 0
                    -13 5 -24 10 -24 15 0 12 60 -2 65 -7 2 -35 6 -61 8 -36 4 -50 1 -54 -10z"/>
                    <path d="M1911 404 c-27 -22 -31 -33 -31 -73 0 -54 26 -87 75 -96 33 -6 96 21
                    103 44 4 11 -6 12 -50 7 -43 -6 -57 -4 -68 8 -14 18 -16 18 69 16 l64 -1 -7
                    33 c-8 43 -57 88 -95 88 -17 0 -43 -11 -60 -26z m87 -26 c20 -20 14 -28 -23
                    -28 -22 0 -35 5 -35 13 0 28 36 37 58 15z"/>
                    <path d="M625 340 c-3 -5 -3 -10 1 -10 4 0 -1 -7 -12 -15 -11 -8 -42 -14 -75
                    -15 -63 0 -82 -14 -94 -67 -6 -29 -4 -33 14 -33 23 0 28 19 10 37 -8 8 -8 17
                    0 32 10 18 21 21 76 21 35 0 67 5 70 10 3 6 15 10 26 10 12 0 19 7 19 20 0 21
                    -24 28 -35 10z"/>
                    <path d="M262 301 c3 -13 14 -15 69 -13 75 4 85 -3 92 -63 l4 -40 1 46 c3 61
                    -15 75 -101 82 -59 4 -68 3 -65 -12z"/>
                    <path d="M650 179 c-197 -55 -210 -59 -193 -66 10 -4 122 24 331 83 28 8 52
                    19 52 24 0 13 15 16 -190 -41z"/>
                    <path d="M40 211 c0 -10 380 -104 388 -96 9 9 0 12 -191 59 -210 52 -197 49
                    -197 37z"/>
                    <path d="M660 144 c-102 -29 -198 -53 -213 -53 -22 -1 -273 58 -404 95 -20 5
                    -23 2 -23 -19 0 -24 7 -26 172 -67 130 -32 177 -48 190 -64 23 -29 93 -29 116
                    1 13 15 60 32 182 64 91 23 169 45 173 49 4 4 4 15 0 26 -8 19 -16 17 -193
                    -32z"/>
                    <path d="M402 163 c4 -21 33 -25 33 -3 0 8 -8 16 -18 18 -14 3 -18 -1 -15 -15z"/>
                    <path d="M920 131 c0 -36 3 -41 24 -41 30 0 49 22 44 52 -2 18 -10 24 -35 26
                    -32 3 -33 2 -33 -37z m49 20 c17 -11 7 -41 -14 -41 -9 0 -15 9 -15 25 0 27 6
                    30 29 16z"/>
                    <path d="M2016 139 c-8 -42 -2 -48 44 -47 30 0 38 3 31 12 -6 7 -8 22 -4 34 3
                    13 1 22 -6 22 -6 0 -11 -7 -11 -15 0 -8 -4 -15 -9 -15 -5 0 -7 9 -4 20 4 15 0
                    20 -15 20 -14 0 -21 -9 -26 -31z"/>
                    <path d="M2188 163 c6 -2 12 -20 12 -39 0 -19 5 -34 10 -34 6 0 10 15 10 34 0
                    19 6 37 13 39 6 3 -4 5 -23 5 -19 0 -29 -2 -22 -5z"/>
                    <path d="M2428 158 c-7 -19 6 -61 19 -65 7 -3 10 8 8 29 -2 18 -4 36 -4 41 -1
                    12 -19 8 -23 -5z"/>
                    <path d="M1025 140 c-11 -18 5 -50 26 -50 10 0 19 5 19 11 0 5 -4 8 -9 4 -5
                    -3 -12 -1 -16 5 -3 6 3 10 15 10 13 0 20 5 18 13 -6 16 -44 21 -53 7z"/>
                    <path d="M1112 123 c2 -20 9 -29 26 -31 12 -2 22 2 22 8 0 7 -7 10 -15 7 -8
                    -4 -17 -2 -20 3 -4 6 3 10 15 10 28 0 21 24 -9 28 -19 3 -22 0 -19 -25z"/>
                    <path d="M1190 106 c0 -25 5 -46 10 -46 6 0 10 6 10 14 0 7 9 16 21 19 32 9
                    22 51 -13 55 -27 3 -28 2 -28 -42z"/>
                    <path d="M1397 136 c-4 -9 -4 -23 -1 -31 7 -17 54 -21 54 -4 0 6 -7 9 -15 6
                    -8 -4 -17 -2 -20 3 -4 6 3 10 15 10 27 0 21 24 -7 28 -11 2 -23 -4 -26 -12z"/>
                    <path d="M1483 120 c-2 -26 1 -30 22 -30 21 0 25 5 25 30 0 24 -4 30 -22 30
                    -18 0 -23 -6 -25 -30z"/>
                    <path d="M1560 120 c0 -16 5 -30 10 -30 6 0 10 8 10 18 0 10 5 23 12 30 9 9 7
                    12 -10 12 -18 0 -22 -6 -22 -30z"/>
                    <path d="M1631 118 c1 -28 2 -30 7 -10 8 28 28 29 25 1 -2 -10 2 -19 7 -19 6
                    0 10 14 10 30 0 25 -4 30 -25 30 -22 0 -25 -4 -24 -32z"/>
                    <path d="M1771 118 c1 -28 2 -30 7 -10 9 30 25 30 34 0 5 -20 6 -18 7 10 1 28
                    -2 32 -24 32 -22 0 -25 -4 -24 -32z"/>
                    <path d="M1857 143 c-15 -15 -6 -45 16 -51 22 -6 22 -6 -3 -13 -22 -5 -23 -7
                    -7 -14 26 -11 47 12 47 51 0 29 -3 34 -23 34 -13 0 -27 -3 -30 -7z"/>
                    <path d="M2274 135 c-4 -8 -4 -22 0 -30 6 -17 46 -21 46 -4 0 6 -7 9 -15 6 -8
                    -4 -17 -2 -20 3 -4 6 3 10 14 10 12 0 21 7 21 15 0 8 -9 15 -20 15 -11 0 -23
                    -7 -26 -15z"/>
                    <path d="M2354 119 c-1 -30 10 -39 19 -16 4 10 6 10 6 0 2 -27 18 -11 17 16
                    -1 21 -6 28 -21 28 -15 0 -20 -7 -21 -28z"/>
                    <path d="M2490 120 c0 -16 5 -30 10 -30 6 0 10 11 10 25 0 14 5 25 10 25 6 0
                    10 -11 10 -25 0 -14 4 -25 9 -25 5 0 7 8 4 19 -3 10 0 22 6 25 7 5 11 -3 11
                    -18 0 -14 5 -26 10 -26 6 0 10 14 10 30 0 29 -2 30 -45 30 -43 0 -45 -1 -45
                    -30z"/>
                    <path d="M2610 120 c0 -16 5 -30 10 -30 6 0 10 14 10 30 0 17 -4 30 -10 30 -5
                    0 -10 -13 -10 -30z"/>
                    <path d="M2671 118 c1 -22 3 -26 6 -13 8 34 23 41 23 12 0 -15 5 -27 10 -27 6
                    0 10 14 10 30 0 25 -4 30 -25 30 -22 0 -25 -4 -24 -32z"/>
                    <path d="M2750 120 c0 -16 5 -30 10 -30 6 0 10 14 10 30 0 17 -4 30 -10 30 -5
                    0 -10 -13 -10 -30z"/>
                    <path d="M2811 118 c1 -22 3 -26 6 -13 8 34 23 41 23 12 0 -15 5 -27 10 -27 6
                    0 10 14 10 30 0 25 -4 30 -25 30 -22 0 -25 -4 -24 -32z"/>
                    <path d="M2897 143 c-15 -15 -6 -45 16 -51 22 -6 22 -6 -3 -13 -22 -5 -23 -7
                    -7 -14 26 -11 47 12 47 51 0 29 -3 34 -23 34 -13 0 -27 -3 -30 -7z"/>
                    </g>
                </svg>                
                
            </div>
            <div>
                <h2>Github Link 👇 </h2> <a href='{link}'>GitHub Link</a>

            </div>
        </div>"""
        )

    document.submit(generate_card_news, [state, document], [state, gallery])
    
    # txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot], scroll_to_output=True)
    # txt.submit(lambda: "", None, txt, scroll_to_output=True)

# share=True : 외부 IP에서 접근 가능한 public link 생성
demo.launch(debug=True, 
            server_name="0.0.0.0", 
            server_port=7777,
            # share=True
            )
