from flask import Flask, render_template, request, jsonify
from chatbot import UniversityQnABot

app = Flask(__name__)
bot = UniversityQnABot("data/university_faqs.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    reply = bot.get_response(user_input)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)
