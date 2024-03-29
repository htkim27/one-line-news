# One-line-news
짧디 짧은 인터넷 기사조차 읽기 어려운 바쁜 현대인들.
그들을 위한 예쁜 카드 뉴스 생성기

## Dependency
Python Version : 3.10.10 
(다른 건 테스트를 안해 봤습니다.)

## Install
```bash
git clone https://github.com/htkim27/one-line-news.git
conda create -n one-line
conda activate one-line
pip install -r requirements.txt
```

[Optional]
```bash
pip install xformers
```

## Run
```bash
python one_line_card_news.py
```

## One Line News (Only text)
```bash
python one_line_card_news_only_text.py
```


## Architecture
![Architecture](./image/architecture.png)

## Gradio (Temp)
- 한 줄 뉴스 생성기 (텍스트만)
https://8ef6cb4ce33c1acbae.gradio.live

- 카드뉴스 생성기
https://f14e060f72a06c07bb.gradio.live

## Hugging Face Model
- 한 줄 뉴스 생성기 : https://huggingface.co/htkim27/one-line-news-v1.2
- 자동 프롬프트 생성기 : https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2
- STABLE_DIFFUSION : https://huggingface.co/stabilityai/stable-diffusion-2-1

## Examples
- 원하던 모습 😊

![Example](./image/example.png)

- 실제 예시 ㅠ

![Example2](./image/example_2.png)

## Reference
- Treform - github : https://github.com/MinSong2/treform
- ployglot-ko-1.3b : https://huggingface.co/EleutherAI/polyglot-ko-1.3b
- KoAlpaca - github : https://github.com/Beomi/KoAlpaca