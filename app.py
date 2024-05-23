from flask import Flask, render_template, request, jsonify
import vertexai
from vertexai.language_models import ChatModel
import os

app = Flask(__name__)
PROJECT_ID = "vulcanv2"  
LOCATION = "us-central1"  

vertexai.init(project=PROJECT_ID, location=LOCATION)

def create_session():
    chat_model = ChatModel.from_pretrained("chat-bison@002")
    chat = chat_model.start_chat()
    return chat

def response(chat, message):
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0.9,
        "top_p": 1
    }
    result = chat.send_message(message, **parameters)
    return result.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/palm2', methods=['GET', 'POST'])
def vertex_palm():
    user_input = ""
    if request.method == 'GET':
        user_input = request.args.get('user_input')
    else:
        user_input = request.form['user_input']
    chat_model = create_session()
    content = response(chat_model,user_input)
    return jsonify(content=content)

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
