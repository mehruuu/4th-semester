from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# Function to get recipes from the API
def get_recipes(query):
    api_key = 'YOUR_API_KEY'  # Replace with your actual API key
    url = f"https://api.spoonacular.com/recipes/complexSearch?query={query}&apiKey={api_key}"
    response = requests.get(url)
    return response.json()

@app.route('/', methods=['GET', 'POST'])
def home():
    recipes = []
    error_message = ''
    if request.method == 'POST':
        query = request.form['query']
        data = get_recipes(query)
        if 'results' in data and len(data['results']) > 0:
            recipes = data['results']
        else:
            error_message = "No recipes found. Try another search."
    return render_template('index.html', recipes=recipes, error_message=error_message)

# Route to display detailed recipe information
@app.route('/recipe/<int:recipe_id>')
def recipe_details(recipe_id):
    api_key = 'YOUR_API_KEY'  # Replace with your actual API key
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={api_key}"
    response = requests.get(url)
    recipe = response.json()
    return render_template('recipe_details.html', recipe=recipe)

if __name__ == "__main__":
    app.run(debug=True)



