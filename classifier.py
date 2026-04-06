import google.genai as genai
import json


def classify_ingredients(api_key, ingredients_text):
    client = genai.Client(api_key=api_key)

    model = "gemini-2.5-flash"

    prompt = f"""
You are a strict JSON generator for dietary classification.

Task:
Classify each ingredient into ONE of these categories:
- Vegan: Contains no animal products (meat, dairy, eggs, honey, etc.)
- Gluten-Free: Naturally free from gluten (fruits, vegetables, meat, dairy, legumes, etc.)
- Keto: Low-carb, high-fat foods that fit a ketogenic diet
- Non-Compliant: Contains gluten, high in carbs, or unsuitable for restrictive diets

Also detect allergens:
- Nuts (tree nuts and peanuts)
- Dairy (milk, cheese, butter, cream, etc.)
- Soy (tofu, soy sauce, edamame, etc.)
- Shellfish (shrimp, crab, lobster, etc.)

Rules:
- Return ONLY valid minified JSON
- No explanations
- No markdown
- No extra text
- Quinoa is Gluten-Free and Keto
- Legumes (beans, lentils) are Vegan and Gluten-Free but NOT Keto

Examples of correct classification:
- "Tofu" -> {{"item":"Tofu","category":"Vegan","allergens":{{"nuts":false,"dairy":false,"soy":true,"shellfish":false}}}}
- "Rice" -> {{"item":"Rice","category":"Gluten-Free","allergens":{{"nuts":false,"dairy":false,"soy":false,"shellfish":false}}}}
- "Avocado" -> {{"item":"Avocado","category":"Keto","allergens":{{"nuts":false,"dairy":false,"soy":false,"shellfish":false}}}}
- "Bread" -> {{"item":"Bread","category":"Non-Compliant","allergens":{{"nuts":false,"dairy":false,"soy":false,"shellfish":false}}}}

Output format:
[
  {{
    "item": "string",
    "category": "Vegan | Gluten-Free | Keto | Non-Compliant",
    "allergens": {{
      "nuts": true/false,
      "dairy": true/false,
      "soy": true/false,
      "shellfish": true/false
    }}
  }}
]

Ingredients:
{ingredients_text}
"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )

        text_output = response.candidates[0].content.parts[0].text.strip()

        # Clean possible markdown
        if text_output.startswith("```"):
            text_output = text_output.strip("```json").strip("```")

        return json.loads(text_output)

    except Exception as e:
        raise RuntimeError(f"LLM Error: {str(e)}")