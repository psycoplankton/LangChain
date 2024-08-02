#Integrate our code with OpenAi API
import os
from constants import openai_key
from langchain_openai import OpenAI 
from langchain import PromptTemplate  # refer https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/
import streamlit as st
from langchain.chains import LLMChain  # refer https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html
from langchain.chains import SimpleSequentialChain, SequentialChain
#SimpleSequentialChain refer https://api.python.langchain.com/en/latest/chains/langchain.chains.sequential.SimpleSequentialChain.html
#SequentialChain refer https://api.python.langchain.com/en/latest/chains/langchain.chains.sequential.SequentialChain.html
from langchain.chains import ConversationBufferMemory
os.environ["OPENAI_API_KEY"] = openai_key

#streamlit framework

st.title('Langchain Demo with OpenAI API')
input_text = st.text_input("search the question you want")
 
#Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template = "Tell me about celebrity {name}"

)


#MEMORY
"""Stores the memory of the inputs given. Just like CahtGPT histroy section"""
person_memory = ConversationBufferMemory(input_key='name' memory='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key = 'chat_history')
description_memory = ConversationBufferMemory(input_key = 'dob', memory_key = 'description_history')


#OPEN AI LLM
llm = OpenAI(temperature=0.8)


#LLM chains
chain = LLMChain(llm=llm, 
                 prompt=first_input_prompt, 
                 verbose=True, 
                 output_keys = 'person')

"""This class is depricated in the newer versions. Instead of these, we use
LangChain runnables. An example code for using runnables is 

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(
    input_variables=["adjective"], template=prompt_template
)
llm = OpenAI()
chain = prompt | llm | StrOutputParser()

chain.invoke("your adjective here")
"""


#Prompt Templates
second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "When was {person} born"

)

chain2 = LLMChain(llm=llm, 
                  prompt=second_input_prompt, 
                  verbose=True, 
                  output_keys='dob')

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = 'Mention 5 major events happened around {dob} in the world'
)

chain3 = LLMChain(llm=llm, 
                  prompt = third_input_prompt, 
                  verbose = True, 
                  output_keys='description')

"""Chains together all the subchains into one."""
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables = ['name'],
    output_variables = ['person', 'dob', 'description'],
    verbose=True
)


"""Simple Sequential Chain only outputs the last text generated"""
SimpleSequentialChain(chains= [chain, chain2])
"""Sequential Chain outputs the whole text"""
SequentialChain(chains= [chain, chain2])


if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(description_memory.buffer)
