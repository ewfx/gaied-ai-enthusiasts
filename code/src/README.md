# Commercial Loan Servicing Assistant

## Description

This project is a Commercial Loan Servicing Assistant built using LangChain and Ollama. It's designed to analyze email content, classify service requests, and extract key data points from commercial loan servicing emails. It utilizes a Large Language Model (LLM) for natural language understanding and a vector database for duplicate detection.

## Features

* **Email Classification:** Classifies service requests in emails into predefined categories and subtypes using **advanced prompts** designed to understand the context and intent of the email content. **It can intelligently identify and categorize multiple service requests within a single email, even if they are expressed in different parts of the email body.**
* **Key Data Extraction:** Extracts relevant data points (e.g., sender, amount, dates) from email content using **dynamic prompts** that adapt to the specific information being requested. 
* **Duplicate Detection:** Identifies duplicate emails using embeddings and a similarity threshold.
* **Flask API:** Provides a REST API for seamless integration with other systems.
* **Ngrok Integration:** Exposes the Flask API publicly using Ngrok for easy access.

## Architecture

The following diagram illustrates the overall architecture of the Commercial Loan Servicing Assistant:

![Architecture Diagram](architecture%20diagram.png)

## Configuration

This project utilizes a configuration file (`configuration.py`) to store important settings and data. 
This file allows you to easily customize the behavior of the assistant.

**Key Configuration Options:**

* `keyMetrics`: A list of key data points that the assistant will try to extract from emails.
* `requestTypesAndSubTypes`:  A dictionary defining the different request categories and subcategories used for email classification.
* `duplicateCheckThresold`: The similarity threshold used to determine if an email is a duplicate. 
* `NGROK_AUTH_TOKEN`: Your Ngrok auth token for exposing the API publicly.

**How to Configure:**

1. **Add your settings:** Define the configuration variables in `configuration.py` in the root directory of your project. according to your needs. 

**Example `configuration.py`:**

## Requirements

* Python 3.7+
* LangChain
* llama3.1:8b
* Ollama
* Flask
* Pyngrok
* Hugging Face Transformers
* FAISS
* eml_parser
* pypdf
* accelerate
* Flask
* pyngrok
* langchain
* langchain_community
* faiss-cpu
* eml_parser
* pypdf
* transformers


## Installation

1. **Clone the repository:**
    git clone https://github.com/ewfx/gaied-ai-enthusiasts/
	
2. **Install dependencies:**
    pip install -r requirements.txt
	
3. **Download Ollama model:**
   !curl -fsSL https://ollama.com/install.sh | sh
   
4. **starting ollama server locally**
   import subprocess
   import time
   process = subprocess.Popen("ollama serve", shell=True)
   time.sleep(5)  # Wait for 5 seconds
	
5. **Download llama3 model using ollama**
   ollama serve ollama pull llama3.1:8b

4. **Configure Ngrok:**
    - Sign up for an free Ngrok account and obtain an auth token from https://dashboard.ngrok.com/get-started/your-authtoken
    - Replace `NGROK_AUTH_TOKEN` in the configuration file
	
## Usage

This project can be run in two ways:

**1. As a Python API:**

* **Run `emailClassification.py`:** This file contains the Flask API implementation. Execute it to start the API server.
* **Access the API:** Use a tool like Postman or `curl` to send requests to the API endpoint `/process_email` with your EML file.

**2. In Google Colab with Free GPU:**

* **Open `emailClassification.ipynb`:** This notebook provides an interactive environment for using the assistant in Google Colab.
* **Enable GPU:** In Colab, navigate to "Runtime" -> "Change runtime type" and select "GPU" as the hardware accelerator. This leverages Colab's free GPU resources for faster processing. 
* **Run the notebook cells:** Execute the cells in the notebook to load the model, process emails, and explore the results.

**Choosing the Right Method:**

* **API:** Ideal for integrating the assistant into existing systems or workflows. Provides programmatic access to its functionalities.
* **Colab:** Suitable for experimentation, testing, and ad-hoc email analysis. Offers the convenience of a pre-configured environment with GPU support.


## Example Request
 curl -X POST -F eml_file=@email.eml https://63d8-34-91-150-31.ngrok-free.app/process_email
 
 ## Response Format
  json {
  "message": "Email processed successfully",
  "results": [
    {
      "email_text": "...", 
      "classification": "...", 
      "duplicate": true,  // or false
      "key_data": {      //key data populated if only duplicate false
        // ... key-value pairs for extracted data
      }
    }
  ]
}
   
