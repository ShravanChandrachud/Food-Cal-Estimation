"""
Ollama Qwen 2.5 3B → structured JSON nutritional summary.
"""

import json
import sqlite3
from ollama import chat
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from rapidfuzz import process, fuzz

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_NUM_CTX,
    USDA_DB_PATH,
    USDA_SEARCH,
    MACRO_NUTRIENT_IDS,
)


class Macros(BaseModel):
    calories_kcal: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: Optional[float] = 0.0


class IngredientEntry(BaseModel):
    name: str
    quantity: str
    weight_grams: float
    macros: Macros


class NutritionalSummary(BaseModel):
    dish_name: str
    ingredients: list[IngredientEntry]
    total_macros: Macros
    notes: Optional[str] = ""


class USDALookup:
    def __init__(self):
        self.conn = None
        self.food_names = []
        self.food_map = {}
        if USDA_DB_PATH.exists():
            self.conn = sqlite3.connect(str(USDA_DB_PATH))
            self.conn.row_factory = sqlite3.Row
            for row in self.conn.execute("SELECT fdc_id, description FROM food"):
                self.food_names.append(row["description"])
                self.food_map[row["description"]] = row["fdc_id"]
            print(f"[USDA] Loaded {len(self.food_names)} foods")

    def lookup(self, ingredient_name):
        if not self.conn:
            return {}
        search = USDA_SEARCH.get(ingredient_name, ingredient_name)
        matches = process.extract(search, self.food_names, scorer=fuzz.WRatio, limit=1)
        if not matches:
            return {}
        best_name, score, _ = matches[0]
        fdc_id = self.food_map[best_name]
        macros = {}
        for row in self.conn.execute(
            "SELECT nutrient_id, amount FROM food_nutrient WHERE fdc_id=?", (fdc_id,)
        ):
            nid = row["nutrient_id"]
            if nid in MACRO_NUTRIENT_IDS:
                macros[MACRO_NUTRIENT_IDS[nid]] = row["amount"]
        return {
            "matched": best_name,
            "fdc_id": fdc_id,
            "score": score,
            "per_100g": macros,
        }

    def close(self):
        if self.conn:
            self.conn.close()


def summarize(ingredient_lines, usda_data=None):
    """
    Send ingredient list + USDA reference to LLM, get JSON macros back.
    Args:
        ingredient_lines: list of strings like "2x egg (~50g each)"
        usda_data: dict {name: {per_100g: {...}}}
    Returns: NutritionalSummary
    """
    usda_ctx = ""
    if usda_data:
        lines = []
        for name, d in usda_data.items():
            m = d.get("per_100g", {})
            if m:
                lines.append(
                    f"  {name}: cal={m.get('calories_kcal', '?')}, "
                    f"prot={m.get('protein_g', '?')}g, "
                    f"fat={m.get('fat_g', '?')}g, "
                    f"carbs={m.get('carbs_g', '?')}g"
                )
        if lines:
            usda_ctx = "\n\nUSDA reference (per 100g):\n" + "\n".join(lines)

    prompt = f"""You are a nutritionist AI. Estimate total macronutrients for this meal.

Detected ingredients:
{chr(10).join(f"- {l}" for l in ingredient_lines)}
{usda_ctx}

Calculate per-ingredient macros scaled to actual weight, then sum for totals.
Infer a dish name from the ingredients."""

    print(f"[LLM] Sending to {OLLAMA_MODEL}...")
    resp = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=NutritionalSummary.model_json_schema(),
        options={"temperature": OLLAMA_TEMPERATURE, "num_ctx": OLLAMA_NUM_CTX},
    )
    return NutritionalSummary.model_validate_json(resp.message.content)


def format_report(s):
    """Pretty-print a NutritionalSummary."""
    lines = [f"{'=' * 50}", f"  {s.dish_name}", f"{'=' * 50}", ""]
    for i in s.ingredients:
        m = i.macros
        lines.append(f"  {i.name} ({i.quantity}, ~{i.weight_grams:.0f}g)")
        lines.append(
            f"    {m.calories_kcal:.0f} kcal | P:{m.protein_g:.1f}g | "
            f"C:{m.carbs_g:.1f}g | F:{m.fat_g:.1f}g"
        )
    t = s.total_macros
    lines += [
        "",
        f"  {'─' * 46}",
        "  TOTAL:",
        f"    Calories: {t.calories_kcal:.0f} kcal",
        f"    Protein:  {t.protein_g:.1f}g",
        f"    Carbs:    {t.carbs_g:.1f}g",
        f"    Fat:      {t.fat_g:.1f}g",
        f"{'=' * 50}",
    ]
    if s.notes:
        lines += ["", f"  Notes: {s.notes}"]
    return "\n".join(lines)
