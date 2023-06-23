import os
import logging

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from cloudevents.http import CloudEvent
from cloudevents.conversion import to_binary
import requests

from flask import Flask, request, jsonify

model_name = 'Hate-speech-CNERG/english-abusive-MuRIL'

TRANSFORMERS_CACHE = os.environ['TRANSFORMERS_CACHE']
moderated_reviews_sink = os.environ['moderated_reviews_sink']
denied_reviews_sink = os.environ['denied_reviews_sink']
attributes = {
    "type": os.environ['ce_type'],
    "source": os.environ['ce_source']
}

def create_app():
    global tokenizer
    global model
    global device

    app = Flask(__name__)
    app.logger.setLevel(logging.INFO)

    app.logger.info("starting app")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    app.logger.info("Model loaded")

    return app

app = create_app()

@app.route("/status")
def status():
    return jsonify({"status": "ok"}), 200

@app.route('/analyze', methods = ['POST'])
def process():
    json_payload = request.json

    try:
        review_text = json_payload['review_text']
    except KeyError:
        print("Not valid data input syntax")
        return 'bad request', 400
    inputs = tokenizer(review_text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    inputs = inputs.to(device)

    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
    score = int(predictions.argmax(axis=1)[0]) - 1  # Convert 0-4 to -1-3
    response = f"{'Non-Abusive' if score < 0 else 'Abusive'}"

    sentiment_data = json_payload
    sentiment_data['score'] = score
    sentiment_data['response'] = response

    event = CloudEvent(attributes, sentiment_data)
    headers, body = to_binary(event)

    if score == 0:
        requests.post(denied_reviews_sink, data=body, headers=headers)
    else :
        requests.post(moderated_reviews_sink, data=body, headers=headers)

    return "Success", 200
  