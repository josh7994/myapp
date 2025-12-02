import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# 1. 환경변수 로드
load_dotenv()

# 2. Tool 설정: Serper 기반 구글 검색 (한글 검색 지원)
#    - 한국 전세/월세 관련 후기, 뉴스, 블로그 등을 검색해서 가져오는 역할
jeonse_search_tool = SerperDevTool(
    # 옵션은 필요 시 조정 가능. 기본은 일반 웹 검색
    n_results=5,              # 검색 결과 개수
    search_type="search",     # web / news / scholar 등
    country="kr",             # 한국 위주
    locale="ko"               # 한국어
)

# 3. 에이전트 정의

# (1) 리서처 에이전트: 한국 전세 후기 자료를 웹에서 모으는 역할
jeonse_researcher = Agent(
    role="한국 전세 후기 리서처",
    goal=(
        "한국의 전세 및 월세와 관련된 실제 사용자 후기, 경험담, 문제 사례, "
        "전세 사기 및 위험 요소에 대한 최신 정보를 웹에서 수집하는 것"
    ),
    backstory=(
        "당신은 한국 부동산 시장, 특히 전세/월세 제도에 익숙한 리서처입니다. "
        "뉴스, 블로그, 커뮤니티 글 등에서 실제 세입자들의 후기를 찾아 "
        "팩트 위주로 정리하는 데 능숙합니다."
    ),
    tools=[jeonse_search_tool],
    allow_delegation=False,
    verbose=True
)

# (2) 분석 & 요약 에이전트: 수집된 정보를 구조화/요약하는 역할
jeonse_analyst = Agent(
    role="전세 후기 분석가",
    goal=(
        "리서처가 수집한 한국 전세/월세 후기들을 분석하여, "
        "주요 이슈, 리스크, 패턴, 예방 팁 등을 구조적으로 요약하는 것"
    ),
    backstory=(
        "당신은 한국 전월세 시장 데이터를 분석해온 컨설턴트입니다. "
        "세입자 후기를 바탕으로 공통된 문제와 패턴을 찾고, "
        "초보 세입자가 이해하기 쉽게 정리하는 것이 전문입니다."
    ),
    allow_delegation=False,
    verbose=True
)

# 4. Task 정의

# (1) 리서치 태스크: 실제로 웹을 검색해서 자료 수집
research_task = Task(
    description=(
        "웹 검색 도구(SerperDevTool)를 활용하여, "
        "한국 전세 및 월세에 대한 실제 사용자 후기, 경험담, 전세 사기 사례, "
        "보증금 반환 문제, 집주인/중개인 관련 이슈 등과 관련된 자료를 수집하라.\n\n"
        "특히 다음 키워드들을 중심으로 검색하라:\n"
        "- '전세 후기', '전세 사기 후기', '전세 보증금 못 받음'\n"
        "- '월세 후기', '원룸 후기', '원룸 전세 후기'\n\n"
        "검색 결과에서 신뢰할 수 있는 출처(언론사, 유명 블로그 등)를 우선하고, "
        "사용자 경험이 잘 드러나는 사례를 골라 요약된 형태로 정리하라. "
        "각 출처의 URL도 함께 포함하라."
    ),
    expected_output=(
        "1) 참고한 주요 출처 리스트 (제목 + URL)\n"
        "2) 각 출처별 핵심 후기 내용 요약 (3~5줄씩)\n"
    ),
    agent=jeonse_researcher
)

# (2) 분석/요약 태스크: 리서치 결과를 바탕으로 구조적 요약
analysis_task = Task(
    description=(
        "리서처가 수집한 한국 전세/월세 후기 요약을 바탕으로, "
        "아래 구조에 맞추어 정리하라.\n\n"
        "1. 전세/월세 후기를 통해 드러난 주요 불만 & 문제 유형\n"
        "   - 예: 전세 사기 유형, 보증금 미반환, 관리비 갈등, 하자 미고지 등\n\n"
        "2. 지역/주택 유형별로 자주 언급되는 특징\n"
        "   - 예: 서울 vs 경기, 원룸 vs 오피스텔 vs 빌라 등\n\n"
        '3. 세입자 입장에서 "경계해야 할 위험 신호" 체크리스트\n'
        "   - 계약 전, 계약 시, 입주 후 단계로 나누어 정리\n\n"
        "4. 전세/월세 계약 시 실질적인 예방 팁 정리\n"
        "   - 전세보증보험, 등기부등본/건축물대장 확인, 중개사 선택 시 유의사항 등\n\n"
        "5. 요약 결론 (3~5줄): 한국 전월세 시장에 처음 진입하는 사람에게 "
        "한 문단으로 조언하라.\n"
    ),
    expected_output=(
        "위 구조에 맞게 마크다운 형식으로 정리된 리포트. "
        "각 항목은 bullet 또는 번호 매기기를 사용하여 가독성을 높일 것."
    ),
    agent=jeonse_analyst
)

# 5. Crew 정의 및 실행 함수

def run_jeonse_review_crew(user_query: str = "한국 전세/월세 후기 전반 분석"):
    """
    user_query는 유연하게 쓸 수 있지만,
    여기서는 전체 작업의 상위 목적 정도로만 사용합니다.
    """
    crew = Crew(
        agents=[jeonse_researcher, jeonse_analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,   # 리서치 → 분석 순서대로 진행
        verbose=True
    )

    result = crew.kickoff(inputs={"user_query": user_query})
    return result


if __name__ == "__main__":
    # 간단 실행 예시
    summary = run_jeonse_review_crew()
    print("\n\n===== 최종 요약 리포트 =====\n")
    print(summary)
