import google.genai as genai
import json
import time
import re


def classify_ingredients(api_key, ingredients_text):
    client = genai.Client(api_key=api_key)

    model = "gemini-2.5-flash"  # Switch to 2.5 flash

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

    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            text_output = response.candidates[0].content.parts[0].text.strip()

            if text_output.startswith("```"):
                text_output = text_output.strip("```json").strip("```")

            return json.loads(text_output)

        except Exception as e:
            error_str = str(e)
            
            # Check for quota exceeded error
            if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
                # Extract retry delay from error message
                retry_match = re.search(r'retry in ([0-9.]+)s', error_str)
                if retry_match:
                    retry_delay = float(retry_match.group(1))
                    print(f"Quota exceeded. Waiting {retry_delay:.1f} seconds...")
                    time.sleep(retry_delay + 1)  # Add 1 second buffer
                    continue
                else:
                    print(f"Quota exceeded. Using fallback classification...")
                    return classify_ingredients_fallback(ingredients_text)
            
            # For other errors, retry with exponential backoff
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Error occurred, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            
            # If all retries failed, use fallback
            print("API unavailable. Using fallback classification...")
            return classify_ingredients_fallback(ingredients_text)
    
    return classify_ingredients_fallback(ingredients_text)


def classify_ingredients_fallback(ingredients_text):
    """Fallback function that provides manual classification when API fails"""
    ingredients = [line.strip() for line in ingredients_text.strip().split('\n') if line.strip()]
    
    # Simple rule-based fallback classifications
    fallback_rules = {
        'almond flour': {'category': 'Keto', 'allergens': {'nuts': True, 'dairy': False, 'soy': False, 'shellfish': False}},
        'whole milk': {'category': 'Gluten-Free', 'allergens': {'nuts': False, 'dairy': True, 'soy': False, 'shellfish': False}},
        'ribeye': {'category': 'Keto', 'allergens': {'nuts': False, 'dairy': False, 'soy': False, 'shellfish': False}},
        'tofu': {'category': 'Vegan', 'allergens': {'nuts': False, 'dairy': False, 'soy': True, 'shellfish': False}},
        'eggs': {'category': 'Keto', 'allergens': {'nuts': False, 'dairy': False, 'soy': False, 'shellfish': False}},
        'shrimp': {'category': 'Keto', 'allergens': {'nuts': False, 'dairy': False, 'soy': False, 'shellfish': True}},
        'butter': {'category': 'Keto', 'allergens': {'nuts': False, 'dairy': True, 'soy': False, 'shellfish': False}},
        'broccoli': {'category': 'Keto', 'allergens': {'nuts': False, 'dairy': False, 'soy': False, 'shellfish': False}},
        'wheat flour': {'category': 'Non-Compliant', 'allergens': {'nuts': False, 'dairy': False, 'soy': False, 'shellfish': False}},
        'soy sauce': {'category': 'Non-Compliant', 'allergens': {'nuts': False, 'dairy': False, 'soy': True, 'shellfish': False}},
        'chicken breast': {'category': 'Keto', 'allergens': {'nuts': False, 'dairy': False, 'soy': False, 'shellfish': False}},
        'cheese': {'category': 'Keto', 'allergens': {'nuts': False, 'dairy': True, 'soy': False, 'shellfish': False}},
        'peanuts': {'category': 'Vegan', 'allergens': {'nuts': True, 'dairy': False, 'soy': False, 'shellfish': False}},
        'salmon': {'category': 'Keto', 'allergens': {'nuts': False, 'dairy': False, 'soy': False, 'shellfish': False}},
        'quinoa': {'category': 'Gluten-Free', 'allergens': {'nuts': False, 'dairy': False, 'soy': False, 'shellfish': False}}
    }
    
    results = []
    for ingredient in ingredients:
        # Clean ingredient name (remove quantities, etc.)
        clean_name = re.sub(r'^\d+\w*\s+', '', ingredient.lower()).strip()
        
        # Find matching rule
        matched_rule = None
        for key, rule in fallback_rules.items():
            if key in clean_name:
                matched_rule = rule
                break
        
        if matched_rule:
            results.append({
                'item': clean_name.title(),
                'category': matched_rule['category'],
                'allergens': matched_rule['allergens']
            })
        else:
            # Default classification for unknown ingredients
            results.append({
                'item': clean_name.title(),
                'category': 'Non-Compliant',
                'allergens': {'nuts': False, 'dairy': False, 'soy': False, 'shellfish': False}
            })
    
    return results