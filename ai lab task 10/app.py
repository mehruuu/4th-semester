from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def chatbot_response(user_input):
    user_input = user_input.lower()
    if "admission" in user_input:
        return "Admissions are open for Fall 2025. Apply via the university website."
    elif "deadline" in user_input:
        return "The application deadline is June 15, 2025."
    elif "requirements" in user_input:
        return "You need at least a 3.0 GPA and English proficiency test results."
    else:
        return "I'm not sure about that. Ask about admissions, deadlines, or requirements."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    reply = chatbot_response(user_input)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)
