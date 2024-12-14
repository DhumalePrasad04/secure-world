from flask import Flask, render_template, request, jsonify
from ollama import chat
import requests

# Initialize the Flask app
app = Flask(__name__)

# Initialize the Ollama client
def respond(question, stock_data):
    # Construct a message for the LLM model that includes both the question and the relevant stock data
    stock_data_str = f"Current stock data: RSI={stock_data.get('RSI', 'N/A')}, MACD={stock_data.get('MACD', 'N/A')}, Signal={stock_data.get('Signal', 'N/A')}"
    response = chat(model='llama3.2:1b', messages=[
        {"role": "user", "content": f"Your task is to analyze stock data. Question: {question}. {stock_data_str}"}
    ])
    return response['message']['content']

# Function to fetch stock data (from Streamlit or an API)
def fetch_stock_data():
    try:
        response = requests.get("http://127.0.0.1:8501/api/stock_data")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to fetch stock data, Status Code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# Route for the index page, rendering the chatbot interface
@app.route("/")
def index():
    return render_template("chatbot.html")

# Route to handle the message generation from the chatbot
@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Get the input message from the request
        user_message = request.json.get("message")
        stock_data = fetch_stock_data()  # Fetch stock data from Streamlit or API
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        if "study stock data" in user_message.lower():
            # If the user is asking to study the stock data, use the stock data in the response
            if stock_data and "error" not in stock_data:
                llm_response = respond(user_message, stock_data)
            else:
                llm_response = "Sorry, I couldn't fetch the stock data."
        else:
            # If it's a general question, just generate a response using the LLM model
            llm_response = respond(user_message, stock_data)

        # Clean response (optional)
        llm_response = llm_response.replace("*", "")

        # Return the response as JSON
        return jsonify({"response": llm_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app on localhost
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
