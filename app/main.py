"""
Copyright ©2024. The Regents of the University of California (Regents). All Rights Reserved.

Permission to use, copy, modify, and distribute this software and its documentation
for educational, research, and not-for-profit purposes, without fee and without a
signed licensing agreement, is hereby granted, provided that the above copyright
notice, this paragraph and the following two paragraphs appear in all copies,
modifications, and distributions.

Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue,
Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE
SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED
"AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS.
"""

#Imports
import boto3
import json
import logging
import os
import streamlit as st
import sys

# Ensure the root folder is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_aws import BedrockLLM, AmazonKnowledgeBasesRetriever
from app.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def assume_role(config):
    # Assumes an IAM role and returns a boto3 session.
    try:
        session = boto3.Session(region_name=config["REGION"])
        sts_client = session.client("sts")
        assumed_role_object = sts_client.assume_role(
            RoleArn=config["ROLE_ARN"],
            RoleSessionName="AssumeRoleSession1"
        )
        credentials = assumed_role_object['Credentials']
        return boto3.Session(
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            region_name=config["REGION"]
        )
    except Exception as e:
        st.error(f"AWS Authentication Error: {e}")
        logger.error(f"AWS Authentication Error: {e}")
        st.stop()
        return None

def initialize_retriever(session, config):
    # Initializes the Amazon Knowledge Bases Retriever.
    try:
        return AmazonKnowledgeBasesRetriever(
            knowledge_base_id=config["KB_NAME"],
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
            region_name=config["REGION"],
            client=session.client("bedrock-agent-runtime"),
        )
    except Exception as e:
        st.error(f"Knowledge Base Retriever Error: {e}")
        logger.error(f"Knowledge Base Retriever Error: {e}")
        st.stop()
        return None

def initialize_llm(session, config):
    # Initializes the Bedrock LLM.
    try:
        model_kwargs_claude = {
            "temperature": 0,
            "top_k": 10,
            "max_tokens": 750,
        }
        return BedrockLLM(
            model_id="anthropic.claude-instant-v1",
            model_kwargs=model_kwargs_claude,
            region_name=config["REGION"],
            client=session.client("bedrock-runtime"),
        )
    except Exception as e:
        st.error(f"Bedrock LLM Initialization Error: {e}")
        logger.error(f"Bedrock LLM Initialization Error: {e}")
        st.stop()
        return None

def create_prompt_templates():
    # Load the prompt templates for the Conversational Retrieval Chain.
    with open("templates/prompt_prefix.txt","r") as file:
        my_template_prefix = file.read()

    with open("templates/prompt_suffix.txt","r") as file:
        my_template_suffix = file.read()

    # Load the few shot examples
    with open("templates/few_shot_examples.json", "r") as file:
        few_shot_examples = json.load(file)

    example_template = """
    Input: {input}
    Output: {output}
    """

    example_prompt = PromptTemplate(input_variables=["input", "output"], template=example_template)

    few_shot_prompt = FewShotPromptTemplate(
        examples=few_shot_examples,
        example_prompt=example_prompt,
        prefix=my_template_prefix,
        suffix=my_template_suffix,
        input_variables=["context", "chat_history", "question"],
    )

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}

    Follow Up Input:
    {question}

    Standalone question:
    """)

    return few_shot_prompt, CONDENSE_QUESTION_PROMPT

def initialize_chat_interface(qa, memory, msgs):
    # Initializes and manages the Streamlit chat interface.
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        with st.spinner("Generating response..."):
            try:
                output = qa.invoke({'question': prompt, 'chat_history': memory.load_memory_variables({})})
                st.chat_message("ai").write(output['answer'])
                logger.info("Response generated successfully.")
                if output['source_documents']:
                    display_source_documents(output['source_documents'])
            except Exception as e:
                st.error(f"Error generating response: {e}")
                logger.error(f"Error generating response: {e}")

    if st.button("Clear Chat History"):
        msgs.clear()
        memory.clear()
        logger.info("Chat history cleared.")
        st.rerun()

def display_source_documents(source_docs):
    # Displays the source documents in the Streamlit app.
    st.subheader("Source Documents")
    for doc in source_docs:
        st.write(f"Source: {doc.metadata.get('kb_url', 'N/A')}")
        st.write(f"Number: {doc.metadata.get('kb_number', 'N/A')}")
        st.write(doc.page_content)
        st.write("---")

def main():
    # Main function to orchestrate the application.
    ConfigManager.initialize()
    config = ConfigManager.config

    session = assume_role(config)
    if session is None:
        return

    st.set_page_config(page_title="bCourses Support ChatBot")
    st.title("RTL Services Support ChatBot")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(
        chat_memory=msgs, memory_key='chat_history', output_key='answer', return_messages=True
    )
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Welcome to RTL Service Support Bot. How can I help you today?")

    retriever = initialize_retriever(session, config)
    if retriever is None:
        return

    llm = initialize_llm(session, config)
    if llm is None:
        return

    few_shot_prompt, condense_question_prompt = create_prompt_templates()

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": few_shot_prompt},
        memory=memory,
        condense_question_prompt=condense_question_prompt,
    )
    logger.info("Conversational Retrieval Chain initialized successfully.")

    initialize_chat_interface(qa, memory, msgs)

if __name__ == "__main__":
    main()