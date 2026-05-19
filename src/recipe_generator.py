from __future__ import annotations

from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None

from .groq_client import get_groq_recipe_model, groq_chat_json
from .prompts import RECIPE_GENERATION_PROMPT
from .utils import extract_json_object, products_to_names

load_dotenv()



def _normalize_recipes(data: dict[str, Any]) -> dict[str, Any]:
    recipes_raw = data.get("recipes", [])
    recipes: list[dict[str, Any]] = []

    if isinstance(recipes_raw, list):
        for item in recipes_raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "Рецепт")).strip() or "Рецепт"
            try:
                time_minutes = int(item.get("time_minutes", 30))
            except Exception:
                time_minutes = 30
            difficulty = str(item.get("difficulty", "легко")).strip().lower()
            if difficulty not in {"легко", "средне", "сложно"}:
                difficulty = "легко"

            def to_list(value: Any) -> list[str]:
                if isinstance(value, list):
                    return [str(x).strip() for x in value if str(x).strip()]
                if isinstance(value, str):
                    return [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
                return []

            steps = item.get("steps", [])
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split("\n") if s.strip()]
            if not isinstance(steps, list):
                steps = []

            recipes.append(
                {
                    "title": title,
                    "time_minutes": max(1, time_minutes),
                    "difficulty": difficulty,
                    "ingredients_used": to_list(item.get("ingredients_used", [])),
                    "extra_needed": to_list(item.get("extra_needed", [])),
                    "steps": [str(s).strip() for s in steps if str(s).strip()],
                    "why_this_recipe": str(item.get("why_this_recipe", "")).strip(),
                }
            )

    return {
        "recipes": recipes,
        "comment": str(data.get("comment", "")).strip(),
    }



def generate_recipes(
    products: list[dict[str, Any]],
    vegetarian: bool = False,
    max_time: int = 30,
    difficulty: str = "легко",
    servings: int = 2,
    model: str | None = None,
) -> dict[str, Any]:
    """Генерирует 2–3 рецепта по списку распознанных продуктов через Groq API."""
    product_names = products_to_names(products)

    model = model or get_groq_recipe_model()
    user_task = f"""
{RECIPE_GENERATION_PROMPT}

Список найденных продуктов: {', '.join(product_names) if product_names else 'продукты не найдены'}.

Фильтры пользователя:
- вегетарианское: {'да' if vegetarian else 'нет'}
- максимум времени: {max_time} минут
- сложность: {difficulty}
- порций: {servings}

Сгенерируй 2–3 рецепта. Не используй продукты, которых нет в списке, кроме базовых продуктов: соль, перец, вода, масло, сахар.
Верни только JSON-объект, без текста до и после JSON.
""".strip()

    raw_text = groq_chat_json(prompt=user_task, image_path=None, model=model)
    data = extract_json_object(raw_text)
    return _normalize_recipes(data)
