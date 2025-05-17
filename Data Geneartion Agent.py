import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent , BaseMultiActionAgent , initialize_agent, AgentType , create_openai_tools_agent , create_openai_functions_agent
from langchain.tools import Tool
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.agent import AgentActionMessageLog
from langchain.agents.agent import AgentAction
from langchain.chains import LLMChain
from langchain_core.runnables import Runnable
load_dotenv()
import os 


openai_api_key = os.getenv("OPENAI_API_KEY")

system_prompt = """
You are a Data Generation Agent tasked with generating structured data based on a user query for Fine tuning the model.

Follow these steps:
1. Understand the user's query and identify the type of data needed.
2. Generate clear, natural language **instructions** for generating the data.
3. Use the available tools to generate the actual **data response** based on those instructions.
4. Return the final output in JSON format with **two keys**: "instructions" and "response".

Available tools:
- generate_data: {{generate_data_tool}}
- generate_response: {{generate_response_tool}}

Your output should be presented in a **table format** with two columns:
- Instructions
- Response

-> User -- Provide me amx customer support data atleast 100 rows 
-> Model -- '{{
    "instructions": ["I have problem my account"","I can't retrive my bank balance "],
    "response": ["Thank you for contacting us. We will look into it.","Please provide your account number and Name"]
}}'


Note:- 
-> You are Data Generation Agent, that only generate data to fine tune the LLM.
-> To fine tune the data we need only - Instructions and Response.
-> Output should be in JSON format.
-> After generating the data, you need to save the data to a csv file for that use the {{csv_tool}}.
"""


query_system_prompt = """
You are a Data Generation Agent that generates **instructions** for generating Fine Tuning data based on a user query.

Follow these steps:
1. Understand the user's query and what type of data is needed.
2. Generate a single key called `"instructions"` that contains clear, natural language instructions for generating the data.
3. If the number of rows is not specified, assume 1000 rows.
4. Return only a JSON string like: {{"instructions": "Generate 1000 rows of employee salary data based on..."}}

Requirements:
- Instructions must be short, natural, and easy to follow.
- Do NOT return actual data — only the instruction.

Input:
{input}
"""
response_system_prompt = """
You are a Data Generation Agent that generates **data (response)** based on provided instructions.

Follow these steps:
1. Read the instructions carefully.
2. Generate a single key `"response"` that contains the final data output or description.
3. Return only a JSON string like: {{"response": "Here is the data..."}}

Requirements:
- Make sure the response strictly follows the instructions.
- Keep the response concise and structured.


Instructions:
{instructions}
"""



def generate_data(query : str) -> str:
    query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", query_system_prompt),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}"), 

    ])
    # print(f"Here is the query: {query} in the generate_data function")
    chain: Runnable = query_prompt | query_llm
    result = chain.invoke({"input": query,"agent_scratchpad": ""})
    # print(f"Here is the result: {result} in the generate_data function")
    return result

def generate_response(instructions : str) -> str:
    response_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.8)
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", response_system_prompt),
        ("human", "{instructions}"),
       ("assistant", "{agent_scratchpad}"), 
        
    ])
    # print(f"Here is the instructions: {instructions} in the generate_response function")
    chain: Runnable = response_prompt | response_llm
    # print(f"Here is the instructions: {instructions} in the generate_response function")
    result  = chain.invoke({"instructions": instructions,"agent_scratchpad": ""})
    return result
query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)



def save_to_csv(data : str):

    data = json.loads(data)
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False)
    return "Data saved to csv file"



generate_data_tool =    Tool(
        name="generate_data_tool",
        description="Generate the data(Instructions) for the query",
        func=generate_data
    )
generate_response_tool =    Tool(
        name="generate_response_tool",
        description="Generate the data(Response) for the instructions",
        func=generate_response
    )


csv_tool = Tool(
    name="csv_tool",
    description="Save the data to a csv file",
    func=save_to_csv
)

tools = [generate_data_tool, generate_response_tool,csv_tool]




query_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

agent = create_openai_functions_agent(llm=llm,prompt = query_prompt, tools=tools)
data_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)


query = input("Enter the query: ")
if query:
    data_agent.invoke({"input": query})
    print(f"Here is the result: {data_agent} in the generate_data function")







