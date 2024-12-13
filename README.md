# bCourses Support ChatBot
Gen AI based bCourses support chatbot POC
This repository contains the code for a Streamlit-based chatbot application using LangChain and AWS Bedrock.

## Features

- Conversational AI powered by Anthropic Claude via Bedrock.
- Retrieval-augmented generation (RAG) using Amazon Knowledge Bases.
- Configuration-based environment management.
- Streamlit interface for easy interaction.

## Setup Instructions

### Clone the Repository:
```
git clone https://github.com/ets-berkeley-edu/bcourses-chatbot-poc.git
cd bcourses-chatbot-poc
```

* Install Python 3
* Create your virtual environment (venv)
* Install dependencies

```
pip3 install -r requirements.txt [--upgrade]
```

### Configure Environments
```
export APP_ENV=development
export APP_LOCAL_CONFIGS=~/Volumes/XYZ/bcourses_chatbot_poc_config
```

### Run Streamlit Application
```
streamlit run app/main.py
```
