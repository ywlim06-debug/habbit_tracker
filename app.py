# app.py
# ─────────────────────────────────────────────────────────────
# AI 습관 트래커 (마법 요정 에디션) - 안정 버전 (~핑 컨셉 강화)
# - OpenAI 리포트 생성(Responses → ChatCompletions fallback)
# - 모델: gpt-5-mini (실패 시 gpt-4o-mini)
# - 체크인/리포트: form으로 안정화
# - 캘린더(월별) + 날짜 상세
# - 물/운동 수치 + 메모(주석)
# - 시각화: 습관별/시간대별 성공률 + 스티커보드(이모지)
# - 날씨 기능 없음
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import calendar
import json
import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:
    alt = None  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =============================
# 기본 설정
# =============================
st.set_page_config(page_title="AI 습관 트래커 (마법 요정)", page_icon="🎮", layout="wide")

APP_TITLE = "🎮 AI 습관 트래커 (마법 요정 에디션)"
PRIMARY_MODEL = "gpt-5-mini"
FALLBACK_MODEL = "gpt-4o-mini"

HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

TIME_SLOTS = [
    ("🌤️", "아침"),
    ("🏙️", "점심"),
    ("🌆", "저녁"),
    ("🌙", "밤"),
]

CITIES = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Ulsan", "Suwon", "Sejong", "Jeju",
]

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]


# =============================
# 유틸
# =============================
def clean(s: str) -> str:
    return (s or "").strip()


def today_iso() -> str:
    return date.today().isoformat()


def pct(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return round(n / d * 100, 1)


def get_record_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["date"]: r for r in records if r.get("date")}


# =============================
# 오리지널 핑 카드 (~핑 컨셉)
# =============================
PING_NAME_POOL = [
    "반짝핑", "용기핑", "집중핑", "다정핑", "수면핑", "정리핑",
    "활력핑", "성장핑", "미소핑", "차분핑", "포근핑", "신나핑",
    "꾸준핑", "도전핑", "절제핑", "햇살핑", "물방울핑", "리듬핑",
]

PING_ELEMENTS = [
    ("💖", "하트"),
    ("✨", "별빛"),
    ("🌿", "초록"),
    ("🌈", "무지개"),
    ("🫧", "버블"),
    ("🎀", "리본"),
]

PING_SPELLS = [
    "반짝반짝 루틴마법, 성공핑!",
    "오늘도 한 걸음, 꾸준핑!",
    "작은 체크가 큰 마법, 반짝핑!",
    "수면 보호막, 포근핑!",
    "집중 레이저, 집중핑!",
    "활력 충전, 활력핑!",
    "마음 안정 주문, 차분핑!",
]

PING_PHRASES = [
    "오늘은 작은 체크 하나가 마법이 될 거핑!",
    "괜찮아핑, 천천히 해도 돼핑. 그래도 계속핑!",
    "너의 리듬을 찾는 중이핑. 이미 잘하고 있핑!",
    "한 번 반짝이면, 내일은 두 번 반짝핑!",
    "지금의 너도 충분히 멋져핑. 다음은 더 좋아질 거핑!",
]


def get_fairy_ping(seed_key: Optional[str] = None) -> Dict[str, Any]:
    rng = random.Random(seed_key or f"{today_iso()}-ping")
    name = rng.choice(PING_NAME_POOL)
    emo, element = rng.choice(PING_ELEMENTS)
    phrase = rng.choice(PING_PHRASES)
    spell = rng.choice(PING_SPELLS)

    stats = {
        "행복💖": rng.randint(40, 95),
        "집중🌟": rng.randint(30, 95),
        "활력💪": rng.randint(30, 95),
        "휴식💤": rng.randint(30, 95),
        "용기🛡️": rng.randint(30, 95),
        "반짝✨": rng.randint(40, 99),
    }

    specialties = [
        "체크박스를 눌러주면 마법봉이 반짝이핑!",
        "물 한 잔마다 반짝 게이지가 차오르핑!",
        "운동하면 활력 스탯이 확 오르핑!",
        "수면 체크하면 포근 보호막이 깔리핑!",
        "공부/독서하면 집중 레벨이 오르핑!",
    ]

    return {
        "name": name,
        "element": element,
        "emoji": emo,
        "phrase": phrase,
        "spell": spell,
        "stats": stats,
        "specialty": rng.choice(specialties),
    }


# =============================
# OpenAI 리포트
# =============================
def _style_system_prompt(style: str) -> str:
    base = (
        "너는 '마법 요정 코치'야. 말투는 귀엽고 친근하게, 문장 끝을 자주 '~핑'으로 마무리해줘. "
        "모든 문장을 ~핑으로 끝내지는 말고 자연스럽게 섞어줘. "
        "의학적/치료적 진단은 하지 말고, 실행 가능한 조언만 해줘. "
        "반드시 출력 형식을 지켜줘."
    )
    if style == "스파르타 코치":
        return base + " 톤은 엄격하고 직설적이핑. 핑계는 컷이핑. 대신 모욕은 절대 금지이핑."
    if style == "따뜻한 멘토":
        return base + " 톤은 따뜻하고 공감적이핑. 작은 성취를 칭찬해주핑. 부담은 줄여주핑."
    return base + " 톤은 게임마스터 같게 퀘스트/레벨업/보상 표현을 섞어주핑."


def _get_openai_client(api_key: str) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않핑. `pip install openai` 해주핑!")
    return OpenAI(api_key=clean(api_key))


def _build_user_prompt(
    mood: int,
    city: str,
    checked_habits: List[str],
    unchecked_habits: List[str],
    water_ml: int,
    exercise_min: int,
    memo: str,
    time_slots_done: List[str],
    ping: Dict[str, Any],
) -> str:
    ping_text = (
        f"{ping.get('emoji')} {ping.get('name')} ({ping.get('element')})\n"
        f"한마디: {ping.get('phrase')}\n"
        f"주문: {ping.get('spell')}\n"
        f"특기: {ping.get('specialty')}\n"
        f"스탯: {ping.get('stats')}"
    )

    return f"""
아래 데이터로 '마법 요정 코치 리포트'를 작성해줘핑.

[도시]
{city}

[오늘 기분 점수]
{mood}/10

[완료한 습관]
{", ".join(checked_habits) if checked_habits else "없음"}

[미완료 습관]
{", ".join(unchecked_habits) if unchecked_habits else "없음"}

[물 마시기]
{water_ml} ml

[운동하기]
{exercise_min} 분

[시간대 체크(완료한 시간대)]
{", ".join(time_slots_done) if time_slots_done else "없음"}

[메모(주석)]
{memo if memo else "(없음)"}

[오늘의 파트너 핑]
{ping_text}

출력 형식(반드시 지켜핑):
## 컨디션 등급
- 등급: (S/A/B/C/D 중 하나)
- 한 줄 요약: (짧게 1문장, ~핑으로 마무리)

## 습관 분석
- 잘한 점: ...
- 아쉬운 점: ...
- 내일 1% 개선: ...

## 내일 미션
- (실행 미션 3개, 아주 구체적이고 작게, 핑 말투를 자연스럽게 섞기)

## 오늘의 파트너 핑
- 핑: (이름/속성)
- 스탯 활용 응원: (스탯 2~3개를 활용해 응원, ~핑 말투)
- 오늘의 주문: (핑의 주문/특기를 참고해 1문장 주문, 반드시 ~핑으로 끝내기)
""".strip()


def _call_openai_responses(client: "OpenAI", model: str, system: str, user: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ],
        temperature=0.75,
    )
    if getattr(resp, "output_text", None):
        return str(resp.output_text).strip()

    out_texts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                out_texts.append(getattr(c, "text", ""))
    text = "\n".join([t for t in out_texts if t]).strip()
    if not text:
        raise RuntimeError("OpenAI 응답에서 텍스트를 추출하지 못했핑.")
    return text


def _call_openai_chat_completions(client: "OpenAI", model: str, system: str, user: str) -> str:
    cc = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.75,
    )
    content = ""
    if cc.choices:
        content = (cc.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Chat Completions 응답이 비어있핑.")
    return content


def generate_report(
    openai_api_key: str,
    coach_style: str,
    mood: int,
    city: str,
    checked_habits: List[str],
    unchecked_habits: List[str],
    water_ml: int,
    exercise_min: int,
    memo: str,
    time_slots_done: List[str],
    ping: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], List[str]]:
    debug: List[str] = []
    api_key = clean(openai_api_key)
    if not api_key:
        return None, "OpenAI API Key가 비어있핑. 사이드바에 입력해주핑!", debug

    system = _style_system_prompt(coach_style)
    user = _build_user_prompt(mood, city, checked_habits, unchecked_habits, water_ml, exercise_min, memo, time_slots_done, ping)

    try:
        client = _get_openai_client(api_key)
    except Exception as e:
        return None, str(e), debug

    # Responses API
    if hasattr(client, "responses"):
        for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                debug.append(f"Trying Responses API model={model}")
                return _call_openai_responses(client, model, system, user), None, debug
            except Exception as e:
                debug.append(f"Responses {model} failed: {type(e).__name__}: {e}")

    # Chat Completions fallback
    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            debug.append(f"Trying Chat Completions model={model}")
            return _call_openai_chat_completions(client, model, system, user), None, debug
        except Exception as e:
            debug.append(f"Chat {model} failed: {type(e).__name__}: {e}")

    return None, "리포트를 생성하지 못했핑. 디버그 로그를 확인해주핑!", debug


# =============================
# 기록(세션) 구조
# =============================
def demo_last_6_days() -> List[Dict[str, Any]]:
    rng = random.Random(20260209)
    today = date.today()
    out: List[Dict[str, Any]] = []
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        checked_cnt = rng.randint(1, 5)
        mood = rng.randint(3, 9)
        water = rng.choice([0, 300, 500, 800, 1200, 1500, 2000])
        ex = rng.choice([0, 10, 20, 30, 40, 60, 90])
        slots = [s for _, s in TIME_SLOTS if rng.random() < 0.5]
        habits = {name: (rng.random() < (checked_cnt / 5)) for _, name in HABITS}
        out.append(
            {
                "date": d.isoformat(),
                "mood": mood,
                "water_ml": water,
                "exercise_min": ex,
                "memo": "",
                "time_slots": slots,
                "habits": habits,
            }
        )
    return out


def ensure_state():
    if "records" not in st.session_state:
        st.session_state.records = demo_last_6_days()
    if "last_ping" not in st.session_state:
        st.session_state.last_ping = get_fairy_ping(seed_key=today_iso())

    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_report_error" not in st.session_state:
        st.session_state.last_report_error = None
    if "last_report_debug" not in st.session_state:
        st.session_state.last_report_debug = []


def upsert_today_record(rec: Dict[str, Any]):
    records: List[Dict[str, Any]] = st.session_state.records
    t = today_iso()
    for i, r in enumerate(records):
        if r.get("date") == t:
            records[i] = rec
            break
    else:
        records.append(rec)
    st.session_state.records = sorted(records, key=lambda x: x.get("date", ""))[-120:]


def last_7_days_rate_df() -> pd.DataFrame:
    recs = sorted(st.session_state.records, key=lambda x: x.get("date", ""))[-7:]
    rows = []
    for r in recs:
        habits = r.get("habits") or {}
        checked = sum(1 for _, name in HABITS if habits.get(name))
        rows.append({"date": r.get("date"), "rate": pct(checked, len(HABITS))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date")
    return df


# =============================
# 캘린더 렌더링
# =============================
def month_calendar_dates(year: int, month: int) -> List[List[Optional[date]]]:
    cal = calendar.Calendar(firstweekday=6)  # Sunday start
    weeks: List[List[Optional[date]]] = []
    for week in cal.monthdatescalendar(year, month):
        weeks.append([d if d.month == month else None for d in week])
    return weeks


def day_badge(rec: Optional[Dict[str, Any]]) -> str:
    if not rec:
        return "⬜"
    habits = rec.get("habits") or {}
    checked = sum(1 for _, name in HABITS if habits.get(name))
    r = checked / len(HABITS)
    if r >= 0.8:
        return "💖"
    if r >= 0.6:
        return "✨"
    if r >= 0.4:
        return "🫧"
    if r > 0:
        return "🌧️"
    return "⬜"


def slot_success_emoji(p: float) -> str:
    if p >= 0.85:
        return "🌟🌟🌟🌟🌟"
    if p >= 0.7:
        return "🌟🌟🌟🌟▫️"
    if p >= 0.55:
        return "🌟🌟🌟▫️▫️"
    if p >= 0.35:
        return "🌟🌟▫️▫️▫️"
    if p > 0:
        return "🌟▫️▫️▫️▫️"
    return "▫️▫️▫️▫️▫️"


def habit_success_icon(done: bool, emoji: str) -> str:
    return f"{emoji}✅" if done else f"{emoji}▫️"


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("🔑 OpenAI API Key")
    default_openai = ""
    try:
        default_openai = str(st.secrets.get("OPENAI_API_KEY", ""))  # type: ignore
    except Exception:
        default_openai = ""
    openai_api_key = st.text_input("OpenAI API Key", value=default_openai, type="password")

    st.divider()
    st.caption("오늘의 체크가 반짝 마법이 되핑 ✨")


# =============================
# Main UI
# =============================
ensure_state()

st.title(APP_TITLE)
st.caption("오늘의 작은 체크가 내일의 마법이 되핑 ✨")

tab1, tab2, tab3 = st.tabs(["✅ 체크인", "🗓️ 캘린더", "📊 시각화"])

# ---------------------------------------------------------
# TAB 1: 체크인
# ---------------------------------------------------------
with tab1:
    ping = st.session_state.last_ping

    st.subheader("✅ 오늘 체크인핑")

    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox("🏙️ 도시 선택", options=CITIES, index=0)
    with c2:
        coach_style = st.radio("🧑‍🏫 코치 스타일", options=COACH_STYLES, horizontal=True)

    # 체크인 form
    with st.form("checkin_form", clear_on_submit=False):
        st.markdown("#### 🎀 습관 스티커 붙이기핑")
        left, right = st.columns(2)
        habits_done: Dict[str, bool] = {}
        for i, (emo, name) in enumerate(HABITS):
            with (left if i % 2 == 0 else right):
                habits_done[name] = st.checkbox(f"{emo} {name}", value=False, key=f"habit_{name}")

        mood = st.slider("😊 오늘 기분 점수", 1, 10, 6)

        x1, x2, x3 = st.columns([1, 1, 2])
        with x1:
            water_ml = st.number_input("💧 물(ml)핑", min_value=0, max_value=5000, value=500, step=100)
        with x2:
            exercise_min = st.number_input("🏃 운동(분)핑", min_value=0, max_value=600, value=20, step=5)
        with x3:
            memo = st.text_input("📝 메모(주석)핑", value="", placeholder="예: 물 2L 목표핑! / 하체운동 20분 / 일찍 자기")

        st.markdown("#### ⏰ 오늘 실천 시간대(반짝 타임핑)")
        slot_cols = st.columns(4)
        slot_done: Dict[str, bool] = {}
        for i, (emo, slot) in enumerate(TIME_SLOTS):
            with slot_cols[i]:
                slot_done[slot] = st.checkbox(f"{emo} {slot}", value=False, key=f"slot_{slot}")

        checked_count = sum(1 for v in habits_done.values() if v)
        rate = pct(checked_count, len(HABITS))

        m1, m2, m3 = st.columns(3)
        m1.metric("달성률", f"{rate}%")
        m2.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
        m3.metric("기분", f"{int(mood)}/10")

        save = st.form_submit_button("💾 오늘 기록 저장하기핑", use_container_width=True)

    if save:
        rec = {
            "date": today_iso(),
            "mood": int(mood),
            "water_ml": int(water_ml),
            "exercise_min": int(exercise_min),
            "memo": memo,
            "time_slots": [s for s, v in slot_done.items() if v],
            "habits": habits_done,
        }
        upsert_today_record(rec)
        st.success("저장 완료핑! 캘린더/통계에 반영되핑 ✨")

    st.subheader("📈 최근 7일 달성률핑")
    df7 = last_7_days_rate_df()
    st.bar_chart(df7.set_index("date")[["rate"]])

    st.divider()
    st.subheader("🧠 컨디션 리포트 생성핑")

    # 리포트 form
    with st.form("report_form", clear_on_submit=False):
        st.markdown(f"### 🎀 오늘의 파트너 핑: {ping['emoji']} {ping['name']} ({ping['element']})")
        st.caption(f"{ping['phrase']} / 주문: {ping['spell']} / 특기: {ping['specialty']}")

        stats_df = pd.DataFrame({"stat": list(ping["stats"].keys()), "value": list(ping["stats"].values())})
        if alt is not None:
            chart = (
                alt.Chart(stats_df)
                .mark_bar(color="#e74c3c")
                .encode(
                    x=alt.X("value:Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("stat:N", sort="-x"),
                    tooltip=["stat", "value"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.bar_chart(stats_df.set_index("stat"))

        generate = st.form_submit_button("✨ 컨디션 리포트 생성하기핑", use_container_width=True)

    if generate:
        rec_map = get_record_map(st.session_state.records)
        rec = rec_map.get(today_iso())
        if rec:
            h = rec.get("habits", {})
            mood_r = int(rec.get("mood", 6))
            water_r = int(rec.get("water_ml", 0))
            ex_r = int(rec.get("exercise_min", 0))
            memo_r = str(rec.get("memo", ""))
            slots_r = rec.get("time_slots", []) or []
        else:
            st.warning("오늘 기록 저장이 아직이핑! 지금 화면 입력값으로 리포트를 만들게핑.")
            h = habits_done
            mood_r = int(mood)
            water_r = int(water_ml)
            ex_r = int(exercise_min)
            memo_r = memo
            slots_r = [s for s, v in slot_done.items() if v]

        checked_habits = [k for k, v in h.items() if v]
        unchecked_habits = [k for k, v in h.items() if not v]

        with st.spinner("핑이 리포트 마법을 쓰는 중이핑...✨"):
            report, err, dbg = generate_report(
                openai_api_key=openai_api_key,
                coach_style=coach_style,
                mood=mood_r,
                city=city,
                checked_habits=checked_habits,
                unchecked_habits=unchecked_habits,
                water_ml=water_r,
                exercise_min=ex_r,
                memo=memo_r,
                time_slots_done=slots_r,
                ping=ping,
            )

        st.session_state.last_report = report
        st.session_state.last_report_error = err
        st.session_state.last_report_debug = dbg

    report = st.session_state.last_report
    err = st.session_state.last_report_error
    dbg = st.session_state.last_report_debug

    st.markdown("### 📝 AI 리포트 결과핑")
    if report:
        st.markdown(report)
    else:
        st.info("아직 리포트가 없핑. 버튼을 눌러 생성해주핑!")
        if err:
            st.error(err)

    with st.expander("🔧 리포트 디버그(오류 원인 확인핑)"):
        st.write(dbg if dbg else ["(디버그 로그 없음)"])

    st.markdown("### 🔗 공유용 텍스트핑")
    share_payload = {
        "date": today_iso(),
        "city": city,
        "coach_style": coach_style,
        "ping": ping,
        "report": report,
        "report_error": err,
        "debug": dbg,
    }
    st.code(json.dumps(share_payload, ensure_ascii=False, indent=2), language="json")

    with st.expander("📎 API 안내 / 준비물핑"):
        st.markdown(
            """
**필요한 것**
- OpenAI API Key (리포트 생성용)핑

**리포트가 안 만들어질 때**
- 사이드바에 키가 제대로 들어갔는지 확인해주핑
- `pip install --upgrade openai` 해주핑
- gpt-5-mini가 안 되면 자동으로 gpt-4o-mini로 시도하핑 (디버그에서 확인 가능이핑)
"""
        )

# ---------------------------------------------------------
# TAB 2: 캘린더
# ---------------------------------------------------------
with tab2:
    st.subheader("🗓️ 캘린더 기록 보기핑")

    today = date.today()
    year = st.number_input("연도", min_value=2020, max_value=2100, value=today.year, step=1)
    month = st.number_input("월", min_value=1, max_value=12, value=today.month, step=1)

    rec_map = get_record_map(st.session_state.records)
    weeks = month_calendar_dates(int(year), int(month))

    st.caption("뱃지: 💖(80%↑) ✨(60%↑) 🫧(40%↑) 🌧️(1~39%) ⬜(0%)핑")

    headers = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    hcols = st.columns(7)
    for i, h in enumerate(headers):
        hcols[i].markdown(f"**{h}**")

    for w in weeks:
        cols = st.columns(7)
        for i, d in enumerate(w):
            if d is None:
                cols[i].write(" ")
                continue
            iso = d.isoformat()
            badge = day_badge(rec_map.get(iso))
            cols[i].markdown(f"**{d.day}** {badge}")

    st.divider()
    st.markdown("### 🔍 특정 날짜 상세 보기핑")
    pick = st.date_input("날짜 선택", value=today, key="calendar_pick")
    iso = pick.isoformat()
    rec = rec_map.get(iso)

    if not rec:
        st.info("해당 날짜 기록이 없핑.")
    else:
        habits = rec.get("habits") or {}
        checked = sum(1 for _, name in HABITS if habits.get(name))
        rate = pct(checked, len(HABITS))

        m1, m2, m3 = st.columns(3)
        m1.metric("달성률", f"{rate}%")
        m2.metric("달성 습관", f"{checked}/{len(HABITS)}")
        m3.metric("기분", f"{rec.get('mood', '-')}/10")

        st.markdown("#### ✅ 습관핑")
        st.write(" · ".join([habit_success_icon(bool(habits.get(name)), emo) + f" {name}" for emo, name in HABITS]))

        st.markdown("#### ⏰ 시간대핑")
        slots = rec.get("time_slots") or []
        st.write(", ".join(slots) if slots else "(없음)")

        st.markdown("#### 💧/🏃 수치핑")
        st.write(f"- 물: {rec.get('water_ml', 0)} ml")
        st.write(f"- 운동: {rec.get('exercise_min', 0)} 분")

        st.markdown("#### 📝 메모(주석)핑")
        st.write(rec.get("memo") or "(없음)")

# ---------------------------------------------------------
# TAB 3: 시각화
# ---------------------------------------------------------
with tab3:
    st.subheader("📊 성공률 시각화(이모지)핑")

    recs = sorted(st.session_state.records, key=lambda x: x.get("date", ""))[-14:]
    if not recs:
        st.info("기록이 없핑. 체크인 탭에서 저장해주핑!")
    else:
        st.markdown("### 1) 습관 종류별 성공률 (최근 14일)핑")
        for emo, name in HABITS:
            total = 0
            done = 0
            for r in recs:
                val = (r.get("habits") or {}).get(name)
                if val is None:
                    continue
                total += 1
                if val:
                    done += 1
            p = (done / total) if total else 0.0
            st.write(f"{emo} **{name}** · {slot_success_emoji(p)} ({round(p*100,1)}%)핑")

        st.divider()

        st.markdown("### 2) 시간대별 실천 비율 (최근 14일)핑")
        for emo, slot in TIME_SLOTS:
            total = len(recs)
            done = sum(1 for r in recs if slot in (r.get("time_slots") or []))
            p = (done / total) if total else 0.0
            st.write(f"{emo} **{slot}** · {slot_success_emoji(p)} ({round(p*100,1)}%)핑")

        st.divider()

        st.markdown("### 3) 스티커보드 (날짜 × 습관)핑")
        rows = []
        for r in recs[-10:]:
            row = {"date": r.get("date", "")}
            habits = r.get("habits") or {}
            for emo, name in HABITS:
                row[name] = habit_success_icon(bool(habits.get(name)), emo)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("### 4) 물/운동 트렌드 (최근 14일)핑")
        df2 = pd.DataFrame(
            [{"date": r.get("date"), "water_ml": r.get("water_ml", 0), "exercise_min": r.get("exercise_min", 0)} for r in recs]
        ).sort_values("date")

        cX, cY = st.columns(2)
        with cX:
            st.markdown("#### 💧 물(ml)핑")
            st.line_chart(df2.set_index("date")[["water_ml"]])
        with cY:
            st.markdown("#### 🏃 운동(분)핑")
            st.line_chart(df2.set_index("date")[["exercise_min"]])

st.caption("© AI 습관 트래커 (마법 요정 에디션) — 오늘의 체크가 내일의 마법이 되핑 ✨")
