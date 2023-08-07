import os
import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from streamlit_chat import message
from langchain.experimental.plan_and_execute import load_agent_executor, load_chat_planner
from langchain.agents.tools import Tool
from langchain import LLMMathChain, SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import get_all_tool_names
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_sql_agent

# Setting page title and header
st.set_page_config(page_title="DataBrain", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>DataBrain - Your Data Analyst Copilot</h1>", unsafe_allow_html=True)


llm = OpenAI(openai_api_key="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", temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
db = SQLDatabase.from_uri("sqlite:///FoodAppBusiness.db")
tools = [
    *SQLDatabaseToolkit(db=db,llm=llm).get_tools(),
    # Tool(
    #     name="Calculator",
    #     func=llm_math_chain.run,
    #     description="useful for when you need to answer questions about math"
    # ),
]
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)
# agent = PlanAndExecute(planner=planner, executor=executor, verbose=True, memory=ConversationBufferMemory())
agent = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db,llm=llm),
    verbose=True
)


# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {
            "role": "system", 
            "content": """
            Read the database.
            Create a hypothesis on why customer complaints have gone up based on this schema, Output this hypothesis. 
            Identify the columns needed to find correlation to prove this hypothesis.
            Query the table using aggregate query or database functions to get the values for thes columns.
            Calculate the correlation coefficient from the value of these columns. 
            Explain the validity of these hypotheses"""
        }
    ]


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    response = agent.run(prompt)
    st.session_state['messages'].append({"role": "assistant", "content": response})
    # print(st.session_state['messages'])
    return response


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))