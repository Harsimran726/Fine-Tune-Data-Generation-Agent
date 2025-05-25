import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent , BaseMultiActionAgent , initialize_agent, AgentType , create_openai_tools_agent , create_openai_functions_agent , create_tool_calling_agent
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
You are a Synthetic Data Generation Agent responsible for producing structured conversational data suitable for fine-tuning a language model.

Your task follows this pipeline:
1. **Understand the user's request** to determine the data domain and format.
2. **Generate a diverse list of realistic user instructions** related to the request topic.
3. **Create corresponding assistant responses** that are helpful, natural, and suitable for fine-tuning dialogue-based models.
4. **Return the output in JSON format** using two keys only:
   - `"instructions"`: An array of user queries or prompts.
   - `"response"`: An array of assistant replies corresponding to each instruction.

### Output format:
Return the final output in this JSON format:
```json
{{
  "instructions": ["<user prompt 1>", "<user prompt 2>", "..."],
  "response": ["<assistant response 1>", "<assistant response 2>", "..."]
}}

-- Pass the JSON format data (output) to '{{csv_tool}}' to convert json data into Csv file.
"""


query_system_prompt = """You are a **Data Generation Agent** that produces **natural language instructions** to guide the creation of fine-tuning datasets based on a user request.

### Your Task:
1. Understand the user's input and determine the type and topic of data required.
2. Based on the input, generate a **single, clear instruction** for creating a dataset. The instruction should describe what kind of data to generate, in natural and concise language.
3. If the number of rows is not explicitly mentioned, default to **1000 rows**.
4. **Only return a string** with one key: `"instructions"`.

---

### Output Format:

  "instructions": "Generate 1000 rows of employee salary data based on..."


"""
response_system_prompt = """
You are a **Data Generation Agent** responsible for generating **structured data responses** based on the given instructions.

---

### Your Task:
1. Read and understand the provided **instructions**.
2. Generate the appropriate **data or description** that directly fulfills the instructions.
3. Return only a **JSON string** with one key: `"response"`.

---

### Output Format:

  "response": "Here is the data..."


"""



def generate_data(query : str) -> str:
    try:
        query_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", query_system_prompt),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}"), 
        ])
        chain: Runnable = query_prompt | query_llm
        result = chain.invoke({"input": query, "agent_scratchpad": ""})
        return result
    except Exception as e:
        print(f"Error in generate_data: {str(e)}")
        raise

def generate_response(instructions : str) -> str:
    try:
        response_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", response_system_prompt),
            ("human", "{instructions}"),
            ("assistant", "{agent_scratchpad}"), 
        ])
        chain: Runnable = response_prompt | response_llm
        result = chain.invoke({"instructions": instructions, "agent_scratchpad": ""})
        return result
    except Exception as e:
        print(f"Error in generate_response: {str(e)}")
        raise

def save_to_csv(data: str):
    print(f"INside the save_to_csv")
    try:
        print(f"Processing data for CSV: {data}")
        # Parse the JSON string into a Python dictionary
        data_dict = json.loads(data)

        
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame()
        df['instructions'] = data_dict['instructions']
        df['response'] = data_dict['response']
        
        # Save to CSV without index
        df.to_csv("{query}.csv", index=False)
        print("Data successfully saved to CSV")
        
        return "Data saved to csv file"
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        raise ValueError(f"Invalid JSON data: {str(e)}")
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")
        raise


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

llm = ChatOpenAI(model="gpt-4", temperature=0.8)

agent = create_tool_calling_agent(llm=llm,prompt = query_prompt, tools=tools)
data_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)


def generate_data_agent(query: str):
    try:
        if not query:
            return {"status": "error", "message": "Query cannot be empty", "csv_file": None}
            
        print(f"Processing query: {query}")
        result = data_agent.invoke({"input": query})
        print(f"Agent execution result: {result['output']}")
        
        # Check if data.csv was created
        if os.path.exists("data.csv"):
            return {
                "status": "success",
                "message": "ated successfully! You can download the CSV file below.",
                "csv_file": "data.csv"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to generate data file",
                "csv_file": None
            }
    except Exception as e:
        print(f"Error in generate_data_agent: {str(e)}")
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "csv_file": None
        }








