# app.py
# AI Habit Tracker (Streamlit)
# Features:
# - 5 habit checkboxes (2-column layout) + mood slider + city select + coach style
# - Achievement metrics + 7-day bar chart (6-day demo + today's data), stored in session_state
# - APIs: OpenWeatherMap weather (KR, Celsius), Dog CEO random image + breed
# - OpenAI AI coach report (gpt-5-mini) with style prompts and structured output
# - Weather + dog image cards + report + share text + API 안내 expander

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
import pandas as pd

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

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Sejong",
    "Jeju",
]

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]

MODEL_NAME = "gpt-5-mini"


# -----------------------------
# API helpers
# -----------------------------
def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap current weather
    - Korean language, Celsius
    - timeout=10
    - On failure returns None
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "kr",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        weather = (data.get("weather") or [{}])[0]
        main = data.get("main") or {}
        wind = data.get("wind") or {}

        return {
            "city": city,
            "temp_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "description": weather.get("description"),
            "wind_mps": wind.get("speed"),
        }
    except Exception:
        return None


def _breed_from_dog_url(url: str) -> str:
    # Dog CEO urls often: https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
    # Extract segment after "/breeds/"
    try:
        marker = "/breeds/"
        if marker not in url:
            return "알 수 없음"
        seg = url.split(marker, 1)[1].split("/", 1)[0]  # e.g., "hound-afghan"
        seg = seg.replace("-", " ")
        return seg.strip() if seg.strip() else "알 수 없음"
    except Exception:
        return "알 수 없음"


def get_dog_image() -> Optional[Dict[str, str]]:
    """
    Dog CEO random image
    - timeout=10
    - On failure returns None
    """
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
# OpenAI report
# -----------------------------
def _get_openai_client(api_key: str) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. requirements에 openai를 추가해 주세요.")
    return OpenAI(api_key=api_key)


def _style_system_prompt(style: str) -> str:
    base = (
        "너는 사용자의 습관 체크인 데이터를 바탕으로 '코치 리포트'를 작성한다. "
        "의학적/치료적 진단은 하지 말고, 실천 가능한 제안만 한다. "
        "출력 형식을 반드시 지켜라."
    )

    if style == "스파르타 코치":
        return (
            base
            + " 톤은 엄격하고 직설적이며 군더더기 없이 짧다. 핑계는 받지 않는다. "
            "다만 모욕/비난은 금지하고, 실행 지침을 명확히 준다."
        )
    if style == "따뜻한 멘토":
        return (
            base
            + " 톤은 따뜻하고 공감적이며 다정하다. 작은 성취를 인정하고, 부담을 낮춘다. "
            "현실적인 한 걸음을 제안한다."
        )
    # 게임 마스터
    return (
        base
        + " 톤은 RPG 게임 마스터처럼 재미있고 몰입감 있게 쓴다. "
        "사용자를 '플레이어'로 부르고, 미션/퀘스트/보상 같은 표현을 섞는다."
    )


def generate_report(
    openai_api_key: str,
    habits_checked: List[str],
    habits_unchecked: List[str],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog_breed: Optional[str],
    coach_style: str,
) -> Optional[str]:
    """
    OpenAI Responses API
    Output sections:
    - 컨디션 등급(S~D)
    - 습관 분석
    - 날씨 코멘트
    - 내일 미션
    - 오늘의 한마디
    """
    if not openai_api_key:
        return None

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')} | {weather.get('description')} | "
            f"{weather.get('temp_c')}°C(체감 {weather.get('feels_like_c')}°C) | "
            f"습도 {weather.get('humidity')}% | 바람 {weather.get('wind_mps')}m/s"
        )

    breed_text = dog_breed if dog_breed else "알 수 없음"

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

## 내일 미션
- (체크박스 습관과 연결된 실행 미션 3개, 구체적이고 작게)

## 오늘의 한마디
- (짧고 임팩트 있게 1문장)
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

        # fallback extraction
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
# Session state: records
# -----------------------------
def _init_demo_records() -> List[Dict[str, Any]]:
    """
    Demo last 6 days, deterministic.
    """
    rng = random.Random(20260209)
    today = date.today()
    out: List[Dict[str, Any]] = []
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        checked_count = rng.randint(1, 5)
        mood = rng.randint(3, 9)
        rate = round(checked_count / 5 * 100, 1)
        out.append(
            {
                "date": d.isoformat(),
                "checked_count": checked_count,
                "rate": rate,
                "mood": mood,
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
    if "last_dog" not in st.session_state:
        st.session_state.last_dog = None


def upsert_today_record(checked_count: int, mood: int):
    today_s = date.today().isoformat()
    rate = round(checked_count / 5 * 100, 1)
    rec = {"date": today_s, "checked_count": checked_count, "rate": rate, "mood": mood}

    records: List[Dict[str, Any]] = st.session_state.records
    for i, r in enumerate(records):
        if r.get("date") == today_s:
            records[i] = rec
            break
    else:
        records.append(rec)

    # Keep only last 7 days (by date)
    records_sorted = sorted(records, key=lambda x: x.get("date", ""))
    st.session_state.records = records_sorted[-7:]


# -----------------------------
# Sidebar keys
# -----------------------------
with st.sidebar:
    st.header("🔑 API 키 설정")

    # Optional: allow secrets fallback while still "input fields" exist
    openai_default = ""
    weather_default = ""
    try:
        openai_default = str(st.secrets.get("OPENAI_API_KEY", ""))  # type: ignore
    except Exception:
        openai_default = ""
    try:
        weather_default = str(st.secrets.get("OPENWEATHER_API_KEY", ""))  # type: ignore
    except Exception:
        weather_default = ""

    openai_api_key = st.text_input("OpenAI API Key", value=openai_default, type="password")
    owm_api_key = st.text_input("OpenWeatherMap API Key", value=weather_default, type="password")

    st.caption("팁: Streamlit Cloud는 Secrets에 저장하면 더 편해요.")


# -----------------------------
# Main UI
# -----------------------------
ensure_state()

st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관을 체크하고, AI 코치 리포트로 내일을 준비해요.")


# --- Check-in UI ---
st.subheader("✅ 습관 체크인")

c1, c2 = st.columns(2)

habit_values: Dict[str, bool] = {}
for i, (emoji, name) in enumerate(HABITS):
    target_col = c1 if i % 2 == 0 else c2
    with target_col:
        habit_values[name] = st.checkbox(f"{emoji} {name}", value=False)

mood = st.slider("😊 오늘 기분 점수", min_value=1, max_value=10, value=6)

c3, c4 = st.columns(2)
with c3:
    city = st.selectbox("🏙️ 도시 선택", options=CITIES, index=0)
with c4:
    coach_style = st.radio("🧑‍🏫 코치 스타일", options=COACH_STYLES, horizontal=True)

checked_habits = [name for name, v in habit_values.items() if v]
unchecked_habits = [name for name, v in habit_values.items() if not v]

checked_count = len(checked_habits)
achievement_rate = round(checked_count / len(HABITS) * 100, 1)

# Save today's record into session_state (always keep it synced)
upsert_today_record(checked_count=checked_count, mood=mood)


# --- Metrics ---
st.subheader("📌 오늘 요약")
m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achievement_rate}%")
m2.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
m3.metric("기분", f"{mood}/10")


# --- Chart (7 days: 6 demo + today) ---
st.subheader("📈 최근 7일 달성률")

records = st.session_state.records
df = pd.DataFrame(records)
# Ensure exactly 7 rows: if fewer, pad with blanks (rare)
if not df.empty:
    df = df.sort_values("date")

# Display bar chart for "rate"
chart_df = df.set_index("date")[["rate"]]
st.bar_chart(chart_df)


# -----------------------------
# Report generation
# -----------------------------
st.subheader("🧠 AI 코치 리포트")

btn = st.button("컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    # Fetch weather + dog
    with st.spinner("날씨와 강아지를 불러오는 중..."):
        weather = get_weather(city, owm_api_key)
        dog = get_dog_image()

    st.session_state.last_weather = weather
    st.session_state.last_dog = dog

    with st.spinner("AI 코치가 리포트를 작성하는 중..."):
        report = generate_report(
            openai_api_key=openai_api_key,
            habits_checked=checked_habits,
            habits_unchecked=unchecked_habits,
            mood=mood,
            weather=weather,
            dog_breed=(dog.get("breed") if dog else None),
            coach_style=coach_style,
        )

    st.session_state.last_report = report


# --- Results display ---
weather = st.session_state.last_weather
dog = st.session_state.last_dog
report = st.session_state.last_report

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
        st.warning("날씨 정보를 불러오지 못했어요. (API Key/도시/네트워크를 확인해 주세요)")

with right:
    st.markdown("### 🐶 오늘의 강아지")
    if dog and dog.get("url"):
        st.image(dog["url"], use_container_width=True)
        st.caption(f"품종(추정): {dog.get('breed', '알 수 없음')}")
    else:
        st.warning("강아지 이미지를 불러오지 못했어요. (잠시 후 다시 시도해 주세요)")


st.markdown("### 📝 AI 코치 리포트")
if report:
    st.markdown(report)
else:
    st.caption("아직 리포트가 없어요. 위 버튼을 눌러 생성해보세요.")


# --- Share text ---
st.markdown("### 🔗 공유용 텍스트")
share_text = {
    "date": date.today().isoformat(),
    "city": city,
    "coach_style": coach_style,
    "achievement_rate": achievement_rate,
    "checked_habits": checked_habits,
    "mood": mood,
    "weather": weather,
    "dog": dog,
    "report": report,
}
st.code(json.dumps(share_text, ensure_ascii=False, indent=2), language="json")


# --- API 안내 ---
with st.expander("📎 API 안내 / 준비물"):
    st.markdown(
        """
**필요한 API**
- OpenAI API Key: 리포트 생성용
- OpenWeatherMap API Key: 날씨 표시용 (Current Weather API)

**키가 없으면?**
- 날씨 키가 없으면: 날씨는 표시되지 않지만 앱은 동작해요(리포트에 '날씨 정보 없음'으로 들어감)
- OpenAI 키가 없으면: 리포트 생성이 되지 않아요

**참고**
- OpenWeatherMap은 도시명이 정확해야 해요(Seoul, Busan 등).
- Dog CEO는 무료 공개 API로, 간혹 네트워크 상태에 따라 실패할 수 있어요.
"""
    )

st.caption("© AI 습관 트래커 — 오늘의 작은 체크가 내일을 바꿔요.")
