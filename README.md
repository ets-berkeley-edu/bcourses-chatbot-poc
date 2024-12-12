# bCourses Support ChatBot
Gen AI based bcourses support chatbot POC
This repository contains the code for a Streamlit-based chatbot application using LangChain and AWS Bedrock.

## Features

- Conversational AI powered by Anthropic Claude via Bedrock.
- Retrieval-augmented generation (RAG) using Amazon Knowledge Bases.
- Configuration-based environment management.
- Streamlit interface for easy interaction.

## Setup Instructions

1. Clone the repository:
   git clone https://github.com/ets-berkeley-edu/bcourses-chatbot-poc.git
   cd bcourses-chatbot-poc

2. Install dependencies
   pip install -r requirements.txt

3. Configure environments
   config/development.py for development
   config/development-local.py for local settings

4. Run streamlit application
   streamlit run app/main.py

