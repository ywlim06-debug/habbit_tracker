# app.py
# Streamlit: AI Habit Tracker (Weather 401-friendly + key test + trimming + secrets fallback)
from __future__ import annotations

import json
import random
from calendar import monthrange
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# -----------------------------
# Constants
# -----------------------------
HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

CITY_OPTIONS: Dict[str, Dict[str, str]] = {
    "Seoul": {"q": "Seoul,KR"},
    "Busan": {"q": "Busan,KR"},
    "Incheon": {"q": "Incheon,KR"},
    "Daegu": {"q": "Daegu,KR"},
    "Daejeon": {"q": "Daejeon,KR"},
    "Gwangju": {"q": "Gwangju,KR"},
    "Ulsan": {"q": "Ulsan,KR"},
    "Suwon": {"q": "Suwon,KR"},
    "Sejong": {"q": "Sejong,KR"},
    "Jeju": {"q": "Jeju,KR"},
}

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]
MODEL_NAME = "gpt-5-mini"
HOLIDAY_COUNTRY = "KR"


# -----------------------------
# Small utils
# -----------------------------
def _clean_key(s: str) -> str:
    # 사용자가 복붙할 때 앞뒤 공백/개행이 섞이는 경우가 매우 흔함
    return (s or "").strip()


def _safe_json_message(resp: requests.Response) -> str:
    try:
        if "application/json" in (resp.headers.get("Content-Type") or ""):
            j = resp.json()
            if isinstance(j, dict) and j.get("message"):
                return str(j["message"])
        # fallback: raw text
        t = resp.text.strip()
        return t[:200] if t else "No response body"
    except Exception:
        return "Failed to parse error body"


# -----------------------------
# Weather (OpenWeatherMap) - Geocoding -> Weather
@@ -219,191 +221,301 @@ def _breed_from_dog_url(url: str) -> str:
        marker = "/breeds/"
        if marker not in url:
            return "알 수 없음"
        seg = url.split(marker, 1)[1].split("/", 1)[0]
        seg = seg.replace("-", " ").strip()
        return seg if seg else "알 수 없음"
    except Exception:
        return "알 수 없음"


def get_dog_image() -> Optional[Dict[str, str]]:
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        img_url = data.get("message")
        if not img_url or not isinstance(img_url, str):
            return None
        return {"url": img_url, "breed": _breed_from_dog_url(img_url)}
    except Exception:
        return None


# -----------------------------
# Quote (Quotable)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_quote() -> Optional[Dict[str, str]]:
    try:
        url = "https://api.quotable.io/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        content = data.get("content")
        author = data.get("author")
        if not content:
            return None
        return {"content": str(content), "author": str(author) if author else "Unknown"}
    except Exception:
        return None


# -----------------------------
# Activity (Bored API)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_activity() -> Optional[Dict[str, str]]:
    try:
        url = "https://www.boredapi.com/api/activity"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        activity = data.get("activity")
        activity_type = data.get("type")
        if not activity:
            return None
        return {"activity": str(activity), "type": str(activity_type) if activity_type else "general"}
    except Exception:
        return None


# -----------------------------
# Public holiday (Nager.Date)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def get_holidays(year: int, country_code: str) -> List[Dict[str, Any]]:
    try:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def get_holiday_on(target_date: date, country_code: str) -> Optional[Dict[str, Any]]:
    holidays = get_holidays(target_date.year, country_code)
    target = target_date.isoformat()
    for holiday in holidays:
        if holiday.get("date") == target:
            return holiday
    return None


# -----------------------------
# OpenAI report
# -----------------------------
def _get_openai_client(api_key: str) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. requirements.txt에 openai를 추가해 주세요.")
    return OpenAI(api_key=_clean_key(api_key))


def _style_system_prompt(style: str) -> str:
    base = (
        "너는 사용자의 습관 체크인 데이터를 바탕으로 '코치 리포트'를 작성한다. "
        "의학적/치료적 진단은 하지 말고, 실천 가능한 제안만 한다. "
        "출력 형식을 반드시 지켜라."
    )
    if style == "스파르타 코치":
        return base + " 톤은 엄격하고 직설적. 짧고 명확. 모욕/비난 금지."
    if style == "따뜻한 멘토":
        return base + " 톤은 따뜻하고 공감적. 작은 성취를 인정하고 부담을 낮춘다."
    return base + " 톤은 RPG 게임 마스터. '플레이어', '퀘스트' 같은 표현을 섞어 재미있게."


def generate_report(
    openai_api_key: str,
    habits_checked: List[str],
    habits_unchecked: List[str],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog_breed: Optional[str],
    quote: Optional[Dict[str, str]],
    activity: Optional[Dict[str, str]],
    holiday: Optional[Dict[str, Any]],
    coach_style: str,
) -> Optional[str]:
    openai_api_key = _clean_key(openai_api_key)
    if not openai_api_key:
        return None

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')}({weather.get('country')}) | {weather.get('description')} | "
            f"{weather.get('temp_c')}°C(체감 {weather.get('feels_like_c')}°C) | "
            f"습도 {weather.get('humidity')}% | 바람 {weather.get('wind_mps')}m/s"
        )

    breed_text = dog_breed if dog_breed else "알 수 없음"
    quote_text = f"{quote.get('content')} — {quote.get('author')}" if quote else "인용구 정보 없음"
    activity_text = (
        f"{activity.get('activity')} (type: {activity.get('type')})" if activity else "활동 제안 없음"
    )
    holiday_text = "없음"
    if holiday:
        local_name = holiday.get("localName") or holiday.get("name") or "공휴일"
        holiday_text = f"{local_name} ({holiday.get('name')})"

    user_prompt = f"""
아래 데이터를 기반으로 리포트를 작성해줘.

[오늘 기분 점수]
{mood}/10

[완료한 습관]
{", ".join(habits_checked) if habits_checked else "없음"}

[미완료 습관]
{", ".join(habits_unchecked) if habits_unchecked else "없음"}

[날씨]
{weather_text}

[오늘의 강아지 품종]
{breed_text}

[오늘의 인용구]
{quote_text}

[오늘의 추천 활동]
{activity_text}

[오늘의 공휴일]
{holiday_text}

출력 형식(반드시 지켜):
## 컨디션 등급
- 등급: (S/A/B/C/D 중 하나)
- 한 줄 요약: ...

## 습관 분석
- 잘한 점: ...
- 아쉬운 점: ...
- 내일 1% 개선: ...

## 날씨 코멘트
- ...

## 컨텍스트 연결
- 오늘의 인용구/추천 활동/공휴일 중 1~2가지를 습관과 자연스럽게 연결해 설명

## 내일 미션
- (체크박스 습관과 연결된 실행 미션 3개)

## 오늘의 한마디
- (짧게 1문장)
""".strip()

    try:
        client = _get_openai_client(openai_api_key)
        resp = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": [{"type": "text", "text": _style_system_prompt(coach_style)}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            temperature=0.7,
        )

        if hasattr(resp, "output_text") and resp.output_text:
            return str(resp.output_text).strip()

        out_texts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    out_texts.append(getattr(c, "text", ""))
        text = "\n".join([t for t in out_texts if t]).strip()
        return text if text else None
    except Exception:
        return None


# -----------------------------
# Session state
# -----------------------------
def _init_demo_records() -> List[Dict[str, Any]]:
    rng = random.Random(20260209)
    today = date.today()
    out: List[Dict[str, Any]] = []
    habit_names = [name for _, name in HABITS]
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        checked_count = rng.randint(1, 5)
        habits = rng.sample(habit_names, k=checked_count)
        m = rng.randint(3, 9)
        rate = round(checked_count / len(HABITS) * 100, 1)
        out.append({"date": d.isoformat(), "checked_count": checked_count, "rate": rate, "mood": m})
        out.append(
            {
                "date": d.isoformat(),
                "checked_count": checked_count,
                "rate": rate,
                "mood": m,
                "habits": habits,
            }
        )
    return out


def ensure_state():
    if "records" not in st.session_state:
        st.session_state.records = _init_demo_records()
    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_weather" not in st.session_state:
        st.session_state.last_weather = None
    if "last_weather_debug" not in st.session_state:
        st.session_state.last_weather_debug = None
    if "last_dog" not in st.session_state:
        st.session_state.last_dog = None
    if "last_quote" not in st.session_state:
        st.session_state.last_quote = None
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = None
    if "last_holiday" not in st.session_state:
        st.session_state.last_holiday = None


def upsert_today_record(checked_count: int, mood: int):
def upsert_today_record(checked_count: int, mood: int, checked_habits: List[str]):
    today_s = date.today().isoformat()
    rate = round(checked_count / len(HABITS) * 100, 1)
    rec = {"date": today_s, "checked_count": checked_count, "rate": rate, "mood": mood}
    rec = {
        "date": today_s,
        "checked_count": checked_count,
        "rate": rate,
        "mood": mood,
        "habits": checked_habits,
    }

    records: List[Dict[str, Any]] = st.session_state.records
    for i, r in enumerate(records):
        if r.get("date") == today_s:
            records[i] = rec
            break
    else:
        records.append(rec)

    st.session_state.records = sorted(records, key=lambda x: x.get("date", ""))[-7:]


# -----------------------------
# Sidebar: keys + test
# -----------------------------
with st.sidebar:
    st.header("🔑 API 키 설정")

    # Secrets fallback (배포 시 편의)
    try:
        default_openai = str(st.secrets.get("OPENAI_API_KEY", ""))  # type: ignore
    except Exception:
        default_openai = ""
    try:
        default_owm = str(st.secrets.get("OPENWEATHER_API_KEY", ""))  # type: ignore
@@ -439,132 +551,262 @@ st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관을 체크하고, AI 코치 리포트로 내일을 준비해요.")

st.subheader("✅ 습관 체크인")

c1, c2 = st.columns(2)
habit_values: Dict[str, bool] = {}
for i, (emoji, name) in enumerate(HABITS):
    with (c1 if i % 2 == 0 else c2):
        habit_values[name] = st.checkbox(f"{emoji} {name}", value=False)

mood = st.slider("😊 오늘 기분 점수", 1, 10, 6)

c3, c4 = st.columns(2)
with c3:
    city_label = st.selectbox("🏙️ 도시 선택", options=list(CITY_OPTIONS.keys()), index=0)
with c4:
    coach_style = st.radio("🧑‍🏫 코치 스타일", options=COACH_STYLES, horizontal=True)

checked_habits = [name for name, v in habit_values.items() if v]
unchecked_habits = [name for name, v in habit_values.items() if not v]

checked_count = len(checked_habits)
achievement_rate = round(checked_count / len(HABITS) * 100, 1)

# Keep today's record synced
upsert_today_record(checked_count=checked_count, mood=mood)
upsert_today_record(checked_count=checked_count, mood=mood, checked_habits=checked_habits)

st.subheader("📌 오늘 요약")
m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achievement_rate}%")
m2.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
m3.metric("기분", f"{mood}/10")

st.subheader("📈 최근 7일 달성률")
df = pd.DataFrame(st.session_state.records).sort_values("date")
st.bar_chart(df.set_index("date")[["rate"]])

st.subheader("🗓️ 달력 뷰로 습관 보기")
cal_c1, cal_c2 = st.columns([2, 1])
with cal_c1:
    selected_month = st.date_input(
        "달력 기준 월",
        value=date.today().replace(day=1),
        min_value=date.today().replace(year=date.today().year - 1, day=1),
        max_value=date.today().replace(year=date.today().year + 1, day=1),
    )
with cal_c2:
    selected_day = st.date_input("상세 보기 날짜", value=date.today())


def _rate_color(rate_value: float) -> str:
    if rate_value >= 80:
        return "#2ecc71"
    if rate_value >= 50:
        return "#f1c40f"
    if rate_value > 0:
        return "#e67e22"
    return "#95a5a6"


def build_calendar_html(target_date: date, records: List[Dict[str, Any]]) -> str:
    year = target_date.year
    month = target_date.month
    first_weekday, days_in_month = monthrange(year, month)
    record_map = {r.get("date"): r for r in records}

    header = "".join(
        f"<th style='padding:8px;background:#f5f7fb;border:1px solid #e3e6ef'>{day}</th>"
        for day in ["월", "화", "수", "목", "금", "토", "일"]
    )

    rows = []
    day = 1
    week = [""] * 7
    for i in range(first_weekday):
        week[i] = ""

    while day <= days_in_month:
        weekday = (first_weekday + day - 1) % 7
        record = record_map.get(date(year, month, day).isoformat(), {})
        rate = float(record.get("rate") or 0)
        mood_value = record.get("mood")
        cell = (
            f"<div style='font-weight:600'>{day}</div>"
            f"<div style='color:{_rate_color(rate)}'>달성 {rate:.0f}%</div>"
        )
        if mood_value:
            cell += f"<div style='color:#6c7a89'>기분 {mood_value}/10</div>"
        week[weekday] = f"<td style='padding:8px;border:1px solid #e3e6ef'>{cell}</td>"

        if weekday == 6:
            rows.append("<tr>" + "".join(week) + "</tr>")
            week = [""] * 7
        day += 1

    if any(week):
        for i, cell in enumerate(week):
            if not cell:
                week[i] = "<td style='padding:8px;border:1px solid #e3e6ef;background:#fafafa'></td>"
        rows.append("<tr>" + "".join(week) + "</tr>")

    body = "\n".join(rows)
    return (
        "<table style='width:100%;border-collapse:collapse;text-align:left'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
    )


st.markdown(build_calendar_html(selected_month, st.session_state.records), unsafe_allow_html=True)

detail_record = next(
    (r for r in st.session_state.records if r.get("date") == selected_day.isoformat()), None
)
if detail_record:
    st.info(
        f"**{selected_day.isoformat()} 요약**\n\n"
        f"- 달성 습관: {detail_record.get('checked_count')}개\n"
        f"- 달성률: {detail_record.get('rate')}%\n"
        f"- 기분: {detail_record.get('mood')}/10\n"
        f"- 완료 습관: {', '.join(detail_record.get('habits') or []) or '기록 없음'}"
    )
else:
    st.caption("선택한 날짜에 기록이 없습니다.")

st.subheader("🧠 AI 코치 리포트")
btn = st.button("컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    with st.spinner("날씨와 강아지를 불러오는 중..."):
    with st.spinner("컨텍스트를 불러오는 중..."):
        weather, weather_dbg = get_weather(city_label, owm_api_key)
        dog = get_dog_image()
        quote = get_quote()
        activity = get_activity()
        holiday = get_holiday_on(date.today(), HOLIDAY_COUNTRY)

    st.session_state.last_weather = weather
    st.session_state.last_weather_debug = weather_dbg
    st.session_state.last_dog = dog
    st.session_state.last_quote = quote
    st.session_state.last_activity = activity
    st.session_state.last_holiday = holiday

    with st.spinner("AI 코치가 리포트를 작성하는 중..."):
        report = generate_report(
            openai_api_key=openai_api_key,
            habits_checked=checked_habits,
            habits_unchecked=unchecked_habits,
            mood=mood,
            weather=weather,
            dog_breed=(dog.get("breed") if dog else None),
            quote=quote,
            activity=activity,
            holiday=holiday,
            coach_style=coach_style,
        )
    st.session_state.last_report = report

# Results
weather = st.session_state.last_weather
dog = st.session_state.last_dog
report = st.session_state.last_report
weather_dbg = st.session_state.last_weather_debug
quote = st.session_state.last_quote
activity = st.session_state.last_activity
holiday = st.session_state.last_holiday

left, right = st.columns(2)

with left:
    st.markdown("### 🌦️ 오늘의 날씨")
    if weather:
        st.info(
            f"**{weather.get('city')}**\n\n"
            f"- 상태: {weather.get('description')}\n"
            f"- 기온: {weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C)\n"
            f"- 습도: {weather.get('humidity')}%\n"
            f"- 바람: {weather.get('wind_mps')} m/s"
        )
    else:
        st.warning("날씨 정보를 불러오지 못했어요.")
        st.info(weather_error_hint(weather_dbg or {}))
        with st.expander("🔧 날씨 디버그 상세"):
            st.write(weather_dbg if weather_dbg else {"ok": False, "reason": "no debug"})

with right:
    st.markdown("### 🐶 오늘의 강아지")
    if dog and dog.get("url"):
        st.image(dog["url"], use_container_width=True)
        st.caption(f"품종(추정): {dog.get('breed', '알 수 없음')}")
    else:
        st.warning("강아지 이미지를 불러오지 못했어요. (잠시 후 다시 시도해 주세요)")

st.markdown("### 🌤️ 오늘의 컨텍스트")
context_cols = st.columns(3)
with context_cols[0]:
    st.markdown("**명언**")
    if quote:
        st.write(f"{quote.get('content')}\n\n— {quote.get('author')}")
    else:
        st.caption("명언을 불러오지 못했어요.")
with context_cols[1]:
    st.markdown("**추천 활동**")
    if activity:
        st.write(f"{activity.get('activity')}\n\n유형: {activity.get('type')}")
    else:
        st.caption("활동을 불러오지 못했어요.")
with context_cols[2]:
    st.markdown("**공휴일**")
    if holiday:
        st.write(f"{holiday.get('localName')} ({holiday.get('name')})")
    else:
        st.caption("오늘은 공휴일이 아닙니다.")

st.markdown("### 📝 AI 코치 리포트")
if report:
    st.markdown(report)
else:
    st.caption("아직 리포트가 없어요. 위 버튼을 눌러 생성해보세요. (OpenAI 키 필요)")

st.markdown("### 🔗 공유용 텍스트")
share_text = {
    "date": date.today().isoformat(),
    "city": city_label,
    "coach_style": coach_style,
    "achievement_rate": achievement_rate,
    "checked_habits": checked_habits,
    "mood": mood,
    "weather": weather,
    "dog": dog,
    "quote": quote,
    "activity": activity,
    "holiday": holiday,
    "report": report,
}
st.code(json.dumps(share_text, ensure_ascii=False, indent=2), language="json")

with st.expander("📎 API 안내 / 준비물"):
    st.markdown(
        """
**OpenWeatherMap 401(Invalid API key)일 때**
- 키 오타/공백/줄바꿈이 가장 흔한 원인입니다(이 앱은 자동 trim 처리하지만, 중간에 공백이 섞인 경우는 그대로 실패합니다).
- OpenWeatherMap에서 발급한 키가 맞는지 확인하세요.
- 발급 직후에는 활성화까지 시간이 걸릴 수 있습니다(보통 5~30분).
- 테스트 버튼으로 먼저 확인해보세요.

**OpenAI**
- OpenAI 키가 없으면 리포트 생성이 되지 않습니다.

**Dog CEO**
- 무료 공개 API라 간헐적 실패 가능

**Quotable / Bored API**
- 무료 공개 API라 응답 지연/실패 가능

**공휴일 (Nager.Date)**
- 국가 코드 기준 공휴일 정보를 제공합니다.
"""
    )

st.caption("© AI 습관 트래커 — 오늘의 작은 체크가 내일을 바꿔요.")
