from keyword_modules.textrank import TextRank
import nltk.tokenize

from typing import List, Optional

class TextRankExtractor:
    
    def __init__(self, stopwords_path:Optional[str] = "") -> None:
        
        
        self.inst = TextRank(lang="ko",
                             stopwords=self._stopwords(stopwords_path))
        self.max:int=5
        self.combined_keywords:bool = True
        
        
    def keyword_extract(self, 
                        sents:List[str]):
        sents_ = nltk.tokenize.sent_tokenize(sents)
        for sent in sents_:
            self.inst.build_keywords(sent)
        return self.inst.get_keywords(self.max, self.combined_keywords)
    
    
    def _stopwords(self, stopwords_path:Optional[str]="") -> List[str]:
        stopwords = []
        
        if stopwords_path:
            
            with open(stopwords_path, encoding='utf-8') as file:
                for line in file:
                    line = line.strip() #or some other preprocessing
                    stopwords.append(line) #storing everything in memory!
        else:
            pass
        
        return stopwords
    
if __name__ == "__main__":
    stopwords_path = ""
    keyword_extractor = TextRankExtractor(stopwords_path="")
    
    document = """삼성전자가 반도체 설계에 필수적인 설계자산(IP) 파트너사들과의 협업을 통해 최첨단 IP 포트폴리오를 늘리며 파운드리(반도체 위탁생산) 생태계 구축에 나선다. 이를 통해 압도적인 규모의 IP를 보유하고 있는 파운드리 업계 1위 대만 TSMC에 대한 추격 속도를 높인다는 방침이다.

14일 삼성전자 반도체 뉴스룸에 따르면 삼성전자는 이달 28일(현지 시각) 미국 새너제이(산호세)에서 열리는 삼성파운드리포럼에서 시놉시스와 케이던스, 알파웨이브 등 IP 파트너와의 협력 내용과 최첨단 IP 로드맵 전략을 공개할 예정이다.

반도체 제품은 수많은 IP의 집합체로, 제품 설계에 필요한 IP를 팹리스(fabless·반도체 설계 전문 회사)가 모두 개발할 수 없기 때문에 통상 IP 회사가 특정 IP를 개발해 팹리스, 종합 반도체 회사(IDM), 파운드리 업체에 제공하고, IP 사용에 따른 라이선스 비용을 받는다.

협업에 따라 삼성전자는 공정설계키트(PDK), 설계 방법론(DM) 등 최첨단 IP 개발에 필요한 파운드리 공정 정보를 IP 파트너에 전달하고, IP 파트너들은 삼성전자 파운드리 공정에 최적화된 IP를 개발, 국내외 팹리스 고객에게 제공한다.

이번 협력에는 파운드리 전 응용처에 필요한 핵심 IP가 포함될 예정이다. 삼성전자는 인공지능(AI)과 그래픽처리장치(GPU), 고성능 컴퓨팅(HPC)뿐만 아니라 오토모티브, 모바일 등 전 분야 고객에게 필요한 핵심 IP를 선제적으로 확보해 새로운 팹리스 고객을 유치하고 고객의 개발 지원 역량을 강화한다는 계획이다.

3나노부터 8나노 공정까지 활용할 수 있는 수십여종의 IP가 IP 포트폴리오에 포함된다. 삼성전자는 PCIe 6.0, DDR5·LPDDR5X 등 고속 데이터 입출력을 가능하게 하는 인터페이스 IP와 칩렛(여러 반도체를 하나의 패키지에 넣는 기술) 등 최첨단 패키지용 UCIe 등을 글로벌 IP 파트너와 함께 개발할 계획이다.

이를 토대로 국내외 팹리스 고객은 자신들의 반도체 제품을 생산할 삼성전자 파운드리 공정에 최적화된 IP를 제품 개발 단계에 따라 적기에 활용할 수 있게 된다. 이를 통해 설계 초기 단계부터 오류를 줄이고 시제품 생산과 검증 양산까지의 시간과 비용을 크게 단축할 수 있을 것으로 기대된다.

반도체 칩은 누가 조기에 칩을 개발해 적시에 시장에 출시하느냐에 따라 성패가 좌우되기 때문에 설계 소요 시간을 단축하는 것이 핵심이다. IP는 통상 제품 개발·검증에 최소 2년∼2년 6개월의 기간이 걸리는데, 업계에서는 팹리스가 IP 개발을 IP 파트너에 맡기면 칩 개발부터 양산에 이르는 시간을 기존 약 3년 6개월∼5년에서 1년 6개월∼2년으로 줄일 수 있다고 보고 있다.

한편 삼성전자는 작년 10월 기준 56개 IP 파트너와 함께 4천개 이상의 IP를 제공하고 있다. 2017년 파운드리사업부 출범 이후 IP 파트너와 IP 개수를 지속적으로 늘리며 3배 수준으로 성장했다.

신종신 삼성전자 파운드리사업부 부사장은 “글로벌 IP 파트너 외에 국내 IP 파트너사와의 협력도 확대해 고객의 혁신 제품 개발과 양산을 더 쉽고 빠르게 지원해 나가겠다”고 밝혔다."""
    
    
    keywords = keyword_extractor.keyword_extract(sents = document)
    
    only_keywords = [keyword_tuple[0] for keyword_tuple in keywords]
    
    print(only_keywords)
