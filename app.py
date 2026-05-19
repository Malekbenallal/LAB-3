from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recipe_generator import generate_recipes
from src.utils import save_json
from src.vlm_recognizer import recognize_products

st.set_page_config(page_title="ЛР3: рецепт по фото холодильника", page_icon="🍽️", layout="wide")

st.title("🍽️ ЛР3 — Рецепт по фото холодильника")
st.write(
    "Приложение распознаёт продукты на фотографии с помощью Groq Vision, "
    "а затем генерирует рецепты с помощью LLM через Groq API."
)

with st.sidebar:
    st.header("Фильтры рецепта")
    vegetarian = st.checkbox("Вегетарианское", value=False)
    max_time = st.slider("Максимальное время готовки, минут", 5, 120, 30, step=5)
    difficulty = st.selectbox("Сложность", ["легко", "средне", "сложно"])
    servings = st.number_input("Порций", min_value=1, max_value=10, value=2, step=1)
    st.caption("Нужен действующий GROQ_API_KEY в файле .env")

uploaded = st.file_uploader("Загрузите фото холодильника или продуктовой полки", type=["jpg", "jpeg", "png", "webp", "gif"])

if uploaded is None:
    st.info("Загрузите фото, чтобы начать распознавание продуктов.")
    st.stop()

suffix = Path(uploaded.name).suffix or ".jpg"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    image_path = Path(tmp.name)

left, right = st.columns([1, 1])
with left:
    st.subheader("Исходное фото")
    st.image(str(image_path), use_container_width=True)

with right:
    st.subheader("Результат распознавания")
    if st.button("1) Распознать продукты", type="primary"):
        with st.spinner("Распознаю продукты на фото..."):
            st.session_state["recognition"] = recognize_products(image_path)
            st.session_state.pop("recipes", None)

    recognition = st.session_state.get("recognition")
    if recognition:
        products = recognition.get("products", [])
        if products:
            st.dataframe(pd.DataFrame(products), use_container_width=True)
        else:
            st.warning("Модель не нашла продукты на фото.")
        st.write("**Описание:**", recognition.get("summary", ""))
        missed = recognition.get("possible_missed", [])
        if missed:
            st.write("**Возможные пропуски:**", ", ".join(missed))

st.divider()

if st.session_state.get("recognition"):
    if st.button("2) Сгенерировать рецепты"):
        with st.spinner("Генерирую рецепты..."):
            st.session_state["recipes"] = generate_recipes(
                st.session_state["recognition"].get("products", []),
                vegetarian=vegetarian,
                max_time=max_time,
                difficulty=difficulty,
                servings=servings,
            )

recipes_result = st.session_state.get("recipes")
if recipes_result:
    st.subheader("Предложенные рецепты")
    for recipe in recipes_result.get("recipes", []):
        with st.expander(f"{recipe.get('title', 'Рецепт')} — {recipe.get('time_minutes', '?')} мин", expanded=True):
            st.write("**Сложность:**", recipe.get("difficulty", "-"))
            st.write("**Используются:**", ", ".join(recipe.get("ingredients_used", [])))
            extra = recipe.get("extra_needed", [])
            st.write("**Дополнительно:**", ", ".join(extra) if extra else "не требуется")
            st.write("**Шаги приготовления:**")
            for i, step in enumerate(recipe.get("steps", []), start=1):
                st.write(f"{i}. {step}")
            st.write("**Почему подходит:**", recipe.get("why_this_recipe", ""))

    if recipes_result.get("comment"):
        st.info(recipes_result["comment"])

    if st.button("Сохранить результат в outputs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("outputs") / f"streamlit_result_{timestamp}.json"
        save_json(
            out_path,
            {
                "recognition": st.session_state.get("recognition"),
                "recipes": recipes_result,
                "filters": {
                    "vegetarian": vegetarian,
                    "max_time": max_time,
                    "difficulty": difficulty,
                    "servings": servings,
                },
            },
        )
        st.success(f"Сохранено: {out_path}")
