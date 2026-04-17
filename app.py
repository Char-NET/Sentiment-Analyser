from flask import Flask, request, render_template
from model import predict_sentiment

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        user_input = request.form.get("text")

        if user_input:
            result, confidence = predict_sentiment(user_input)

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)