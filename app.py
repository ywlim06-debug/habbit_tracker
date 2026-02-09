# app.py
# Streamlit: AI Habit Tracker (Weather fix: Geocoding -> lat/lon -> Weather)
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

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

# 도시 선택은 UI용 라벨, 실제 API는 표준 city + country code로 매핑
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


# -----------------------------
# API helpers (FIXED)
# -----------------------------
def _owm_geocode(city_q: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap Geocoding API:
    city_q: "Seoul,KR" 형태 권장
    returns: {"name":..., "lat":..., "lon":..., "country":...} or None
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/geo/1.0/direct"
        params = {"q": city_q, "limit": 1, "appid": api_key}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        arr = r.json()
        if not isinstance(arr, list) or len(arr) == 0:
            return None
        item = arr[0] or {}
        if "lat" not in item or "lon" not in item:
            return None
        return item
    except Exception:
        return None


def get_weather(city_label: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Weather fetch strategy:
    1) Geocode city -> lat/lon
    2) Call weather by lat/lon (Celsius, Korean)
    Failure -> returns None
    """
    if not api_key:
        return None

    city_q = CITY_OPTIONS.get(city_label, {}).get("q", city_label)

    try:
        geo = _owm_geocode(city_q, api_key)
        if not geo:
            return None

        lat, lon = geo["lat"], geo["lon"]

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": lat,
            "lon": lon,
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
            "city": f"{geo.get('name', city_label)}",
            "country": geo.get("country"),
            "temp_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "description": weather.get("description"),
            "wind_mps": wind.get("speed"),
        }
    except Exception:
        return None


def get_weather_debug(city_label: str, api_key: str) -> Dict[str, Any]:
    """
    디버그용: 실패 원인을 UI에 보여주기 위해 status/message를 포함해 반환.
    키는 절대 출력하지 않음.
    """
    if not api_key:
        return {"ok": False, "reason": "OpenWeatherMap API Key가 비어있음"}

    city_q = CITY_OPTIONS.get(city_label, {}).get("q", city_label)

    # 1) geocode
    try:
        geo_url = "https://api.openweathermap.org/geo/1.0/direct"
        geo_params = {"q": city_q, "limit": 1, "appid": api_key}
        gr = requests.get(geo_url, params=geo_params, timeout=10)
        if gr.status_code != 200:
            return {
                "ok": False,
                "step": "geocode",
                "status_code": gr.status_code,
                "message": (gr.json().get("message") if isinstance(gr.json(), dict) else str(gr.text)[:200]),
                "query": city_q,
            }
        arr = gr.json()
        if not isinstance(arr, list) or len(arr) == 0:
            return {"ok": False, "step": "geocode", "reason": "도시 검색 결과 0개", "query": city_q}
        geo = arr[0]
        lat, lon = geo.get("lat"), geo.get("lon")
        if lat is None or lon is None:
            return {"ok": False, "step": "geocode", "reason": "lat/lon 없음", "query": city_q}

        # 2) weather
        w_url = "https://api.openweathermap.org/data/2.5/weather"
        w_params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric", "lang": "kr"}
        wr = requests.get(w_url, params=w_params, timeout=10)
        if wr.status_code != 200:
            j = wr.json() if "application/json" in wr.headers.get("Content-Type", "") else {}
            return {
                "ok": False,
                "step": "weather",
                "status_code": wr.status_code,
                "message": (j.get("message") if isinstance(j, dict) else str(wr.text)[:200]),
                "lat": lat,
                "lon": lon,
            }

        return {"ok": True, "query": city_q, "lat": lat, "lon": lon}
    except Exception as e:
        return {"ok": False, "reason": f"예외 발생: {type(e).__name__}"}


def _breed_from_dog_url(url: str) -> str:
    try:
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
# OpenAI report
# -----------------------------
def _get_openai_client(api_key: str) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. requirements.txt에 openai를 추가해 주세요.")
    return OpenAI(api_key=api_key)


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
    coach_style: str,
) -> Optional[str]:
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
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        checked_count = rng.randint(1, 5)
        mood = rng.randint(3, 9)
        rate = round(checked_count / 5 * 100, 1)
        out.append({"date": d.isoformat(), "checked_count": checked_count, "rate": rate, "mood": mood})
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


def upsert_today_record(checked_count: int, mood: int):
    today_s = date.today().isoformat()
    rate = round(checked_count / len(HABITS) * 100, 1)
    rec = {"date": today_s, "checked_count": checked_count, "rate": rate, "mood": mood}

    records: List[Dict[str, Any]] = st.session_state.records
    for i, r in enumerate(records):
        if r.get("date") == today_s:
            records[i] = rec
            break
    else:
        records.append(rec)

    records_sorted = sorted(records, key=lambda x: x.get("date", ""))
    st.session_state.records = records_sorted[-7:]


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🔑 API 키 설정")
    openai_api_key = st.text_input("OpenAI API Key", value="", type="password")
    owm_api_key = st.text_input("OpenWeatherMap API Key", value="", type="password")
    st.caption("OpenWeatherMap: Geocoding + Weather로 안정적으로 호출합니다.")


# -----------------------------
# Main UI
# -----------------------------
ensure_state()

st.title("📊 AI 습관 트래커")
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

upsert_today_record(checked_count=checked_count, mood=mood)

st.subheader("📌 오늘 요약")
m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achievement_rate}%")
m2.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
m3.metric("기분", f"{mood}/10")

st.subheader("📈 최근 7일 달성률")
df = pd.DataFrame(st.session_state.records).sort_values("date")
st.bar_chart(df.set_index("date")[["rate"]])

st.subheader("🧠 AI 코치 리포트")
btn = st.button("컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    with st.spinner("날씨와 강아지를 불러오는 중..."):
        weather = get_weather(city_label, owm_api_key)
        weather_dbg = get_weather_debug(city_label, owm_api_key)
        dog = get_dog_image()

    st.session_state.last_weather = weather
    st.session_state.last_weather_debug = weather_dbg
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

weather = st.session_state.last_weather
dog = st.session_state.last_dog
report = st.session_state.last_report
weather_dbg = st.session_state.last_weather_debug

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
        with st.expander("🔧 날씨 디버그 정보(원인 확인)"):
            st.write(weather_dbg if weather_dbg else {"ok": False, "reason": "디버그 정보 없음"})

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
    "report": report,
}
st.code(json.dumps(share_text, ensure_ascii=False, indent=2), language="json")

with st.expander("📎 API 안내 / 준비물"):
    st.markdown(
        """
**OpenWeatherMap 날씨가 안 될 때 체크**
- API Key가 맞는지(오타/공백) 확인
- 키가 활성화되어 있는지(발급 직후 5~30분 지연될 수 있음)
- Free 플랜에서도 `Current Weather`와 `Geocoding`은 사용 가능
- 디버그 expander에서 `status_code`가 401이면 키 문제, 404면 도시 검색 문제일 가능성이 큼

**Dog CEO**
- 무료 공개 API라 간헐적 실패 가능

**OpenAI**
- 키가 없으면 리포트 생성 불가
"""
    )

st.caption("© AI 습관 트래커 — 오늘의 작은 체크가 내일을 바꿔요.")

