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
import os
import streamlit as st
import sys

# Ensure the root folder is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Bedrock
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from app.config_manager import ConfigManager

# Initialize configurations
ConfigManager.initialize()
config = ConfigManager.config

#Configure streamlit app
st.set_page_config(page_title="bCourses Support ChatBot")
st.title("bCourses Support ChatBot")

print(f' KB_NAME : {config["KB_NAME"]}')

#Define the retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=config["KB_NAME"],
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
    # credentials_profile_name=config["PROFILE_NAME"],
    region_name=config["REGION"]
)

#Define model parameters
model_kwargs_claude = {
  "temperature" : 0,
  "top_k" : 10,
  "max_tokens_to_sample" : 750
}

#Configure llm
llm = Bedrock(
    model_id="anthropic.claude-instant-v1", 
    model_kwargs=model_kwargs_claude,
    # credentials_profile_name=config["PROFILE_NAME"],
    region_name=config["REGION"]
  )

#Set up message history
msgs = StreamlitChatMessageHistory(key = "langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs, memory_key='chat_history', output_key='answer', return_messages=True)
if len(msgs.messages) == 0:
  msgs.add_ai_message("Welcome to bCourse Support. How can I help you today?")

#Creating the template   
my_template = """
Human: 
    You are a conversational assistant designed to help answer questions from a knowledge base. 
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible. 

{context}

{chat_history}

Question: {question}

Assistant:
"""

#configure prompt template
prompt_template = PromptTemplate(
  input_variables= ['context', 'chat_history', 'question'],
  template= my_template
)

#Configure the chain
qa = ConversationalRetrievalChain.from_llm(
  llm = llm,
  retriever= retriever,
  return_source_documents = True,
  combine_docs_chain_kwargs= {"prompt" : prompt_template},
  memory = memory,
  condense_question_prompt= CONDENSE_QUESTION_PROMPT
)

#Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
  st.chat_message(msg.type).write(msg.content)

#If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
  st.chat_message("human").write(prompt)

  #Invoke the model
  output = qa.invoke({'question' : prompt, 'chat_history' : memory.load_memory_variables({})})
    
  #display the output
  st.chat_message("ai").write(output['answer'])  