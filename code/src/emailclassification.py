#!pip install --upgrade langchain langchain_community faiss-cpu eml_parser pypdf transformers accelerate Flask pyngrok

import os
import json
import torch
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import faiss
import json
import hashlib
import numpy as np
import email
import os
import eml_parser
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from scipy.spatial.distance import cosine
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
import configuration
import email
from bs4 import BeautifulSoup
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from openpyxl import load_workbook
import extractEmailData
import CheckDupliateEmails
import StoreEmails
import extractJsonUtil
from flask import Flask, request, jsonify
import json
from pyngrok import ngrok

# Load LLaMA 3 Model
llm = Ollama(model="llama3.1:8b")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Function to extract key data points dynamically
def extract_key_data(text):
    prompt = f"""
    Extract key data points from the following text:

    Text: "{text}"

    Identify fields such as {configuration.keyMetrics}
    Respond in JSON format like this:
    {{
        "sender_name": "string",
        "amount": "string",
        "expiration_date": "string",
        "date_to_transfer": "string"
    }}
    """

    response = llm.invoke(prompt)
    try:
        extracted_data = extractJsonUtil.extract_json_from_string(response)
    except json.JSONDecodeError:
        extracted_data = {}
    return extracted_data

# Function to classify emails
def classify_email(text):

    prompt = f"""
You are a commercial loan servicing assistant. Your task is to analyze the provided email body and identify all service requests mentioned, classifying each into a specific category and subtype.

Email Body:
{text}

Available Request Types and Subtypes:
{configuration.requestTypesAndSubTypes}

Instructions:
1. Thoroughly review the email body and identify all instances where a service request is being made.
2. For each request, determine the most appropriate request category from the provided list.
3. If applicable, select the relevant subrequest type based on the specifics of the request.
4. Base your confidence on factors like:
    - Presence of keywords related to the request category/subtype.
    - Overall semantic similarity between the request and the category/subtype description.
    - The strength and clarity of the request expressed in the email.
5. If a single sentence or phrase implies multiple requests, prioritize the most specific and explicit request.
6. If there are overlapping or ambiguous requests, use your best judgment to classify them based on the overall context of the email.

Return your classification in the following JSON format:

{{
  "requests": [
    {{
      "request_category": "Selected Category",
      "subrequest_type": "Selected Subtype",
      "confidence_reason": "Brief explanation for your confidence level"
      "confidence score percentage": ""
    }},
    {{
      "request_category": "Selected Category",
      "subrequest_type": "Selected Subtype",
      "confidence_reason": "Brief explanation for your confidence level"
      "confidence score percentage": ""
    }},
    // ... more requests
  ]
}}
"""

    response = llm.invoke(prompt)
    #print("llm res", response)
    try:
        extracted_data = extractJsonUtil.extract_requests_json(response)
    except json.JSONDecodeError:
        extracted_data = []
    return extracted_data

example_texts = ["This is an example text for initialization."]
vectorstore = FAISS.from_texts(example_texts, embedding_model)

all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
print(len(all_docs))

#Function to process eml files
def process_eml_file(eml_file_path):
    """Processes an EML file and returns the results."""
    results = []

    if eml_file_path.endswith(".eml"):
        email_text = extractEmailData.extract_email_data(eml_file_path)
        email_content = email_text['subject'] + email_text['body']
        email_attachments_content = email_text['attachments']

        is_dup, source, score = CheckDupliateEmails.check_duplicate(email_content, vectorstore, embedding_model, configuration.duplicateCheckThresold)

        response_json = {}
        if is_dup:
            response_json = {
                "email_text": email_content[:100] + "...",
                "classification": "duplicate",
                "duplicate": is_dup,
            }
        else:
            classification = classify_email(email_content)
            extracted_data = extract_key_data(email_content)
            StoreEmails.store_email(email_content, vectorstore, embedding_model)
            response_json = {
                "email_text": email_content[:100] + "...",
                "classification": classification,
                "duplicate": is_dup,
                "key_data": extracted_data
            }
        results.append(response_json)

        print("resuls", json.dumps(results, indent=2))

    return results

#testing processing of eml file
process_eml_file("email.eml")

#exposing the API using flask
app = Flask(__name__)

@app.route('/process_email', methods=['POST'])
def process_email():
    print("into process mail")
    if 'eml_file' not in request.files:
        return jsonify({'error': 'No EML file provided'}), 400

    print("eml file exists")
    eml_file = request.files['eml_file']
    eml_file.save('temp_email.eml')  # Save temporarily
    print("saved to temp file")
    results = process_eml_file('temp_email.eml')  # Call the processing method
    print("results received ", results)
    response_data = {
        "message": "Email processed successfully",
        "results": results
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
     ngrok.set_auth_token("2t6u6zBAZFZh2JGODqQ0s0MyQt8_7XXGdkuk8Mm3UhCdw73WA") # signup and get auth token from https://dashboard.ngrok.com/get-started/your-authtoken
     public_url = ngrok.connect(5000).public_url
     print(f" * Running on {public_url} (Press CTRL+C to quit)")
     app.run(host='0.0.0.0', port=5000, debug=True)