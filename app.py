import os
from dotenv import load_dotenv
from classifier import classify_ingredients
from utils import read_ingredients, save_json


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("API key not found in .env")

    try:
        ingredients = read_ingredients("ingredients.txt")

        result = classify_ingredients(api_key, ingredients)

        save_json(result, "categorized_ingredients.json")

        print("Classification completed successfully!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()