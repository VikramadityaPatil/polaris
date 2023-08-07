from langchain.llms import OpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents.tools import Tool
from langchain import LLMMathChain, SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import get_all_tool_names
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_sql_agent


print(get_all_tool_names())

# llm = OpenAI(openai_api_key="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", temperature=0)
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

agent.run("""
Read  my database schema.
Create a hypothesis on which users buy more wine based on this schema, Output this hypothesis. 
Identify the two columns needed to find correlation to prove this hypothesis.
Query the table using aggregate query or database functions to get the values for these two columns.
Calculate the correlation coefficient from the value of these two columns. 
Explain the validity of these hypotheses. 
""")