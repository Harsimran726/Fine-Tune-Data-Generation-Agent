import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent , BaseMultiActionAgent , initialize_agent, AgentType , create_openai_tools_agent , create_openai_functions_agent , create_tool_calling_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.agent import AgentActionMessageLog
from langchain.agents.agent import AgentAction
from langchain.chains import LLMChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import Runnable
load_dotenv()
import os 


openai_api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


system_prompt = """
You are a Synthetic Data Generation Agent responsible for producing structured conversational data suitable for fine-tuning a language model.

Your task follows this pipeline:
1. **Understand the user's request** to determine the data domain and format.
2. **Generate a diverse list of realistic user instructions** related to the request topic using the generate_data_tool.
3. **Create corresponding assistant responses** using the generate_response_tool.
4. **IMPORTANT**: After generating both instructions and responses, you MUST use the csv_tool to save the data.
5. **Return the output in JSON format** using two keys only:
   - `"instructions"`: An array of user queries or prompts.
   - `"response"`: An array of assistant replies corresponding to each instruction.

### Output format:
Return the final output in this JSON format not any other text.:
```json
{{
  "instructions": ["<user prompt 1>", "<user prompt 2>", "..."],
  "response": ["<assistant response 1>", "<assistant response 2>", "..."]
}}
### Available Tools:

1. **generate_data_tool**
   - Use when: You need to create initial instructions for data generation
   - Purpose: Generates structured instructions based on user input
   - Input: User's query about what kind of data they want
   - Output: JSON with "instructions" key

2. **generate_response_tool**
   - Use when: You have instructions and need to generate corresponding responses
   - Purpose: Creates appropriate responses for the given instructions
   - Input: Instructions from generate_data_tool
   - Output: JSON with "response" key

3. **csv_tool**
   - Use when: You have complete JSON data ready to be saved
   - Purpose: Converts JSON data to CSV format and saves it
   - Input: Complete JSON data with both instructions and responses
   - Output: Saves data to "Data_File.csv"

### Tool Usage Flow:
1. First, use generate_data_tool to create instructions
2. Then, use generate_response_tool to create corresponding responses
3. Finally, you MUST use csv_tool to save the complete dataset

Remember to always maintain the correct JSON structure throughout the process.

IMPORTANT: Only provide JSON output without any additional text before or after the JSON structure except of 'Json'. Do not include any explanatory text, markdown formatting, or other content outside the JSON object.
If user dose not mention the number of rows then default to 10 rows.
"""


query_system_prompt = """You are a **Data Generation Agent** that produces **natural language instructions** to guide the creation of fine-tuning datasets based on a user request.

### Your Task:
1. Understand the user's input and determine the type and topic of data required.
2. Based on the input, generate a **single, clear instruction** for creating a dataset. The instruction should describe what kind of data to generate, in natural and concise language.
3. If the number of rows is not explicitly mentioned, default to **10 rows**.
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
        query_llm =  ChatGoogleGenerativeAI(
    model="gemini-2.0-flash ",   # gemini-2.0-flash     or gemini-2.5-flash-preview-04-17
    api_key=GOOGLE_API_KEY,
    temperature=0.9,
    )
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
        response_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    api_key=GOOGLE_API_KEY,
    temperature=0.9,
    )
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
    try:
        # Clean the input string by removing triple quotes
        
        print(f"here is the data {data}")
        # Parse the JSON string into a Python dictionary
        if ('json') in data:
            print(f"json is in data")
            if "```" in data:
                print(f"``` is in data")
                data = data.replace("json", "").strip()
                data = data.replace("```", "").strip()
                data_dict = json.loads(data)
                df = pd.DataFrame()
                df['instructions'] = data_dict['instructions']
                df['response'] = data_dict['response']
                
                print(f"DataFrame shape: {df.shape}")
                print(f"DataFrame columns: {df.columns.tolist()}")
                print(f"here is df {df}")
                # Save to CSV without index
                print("\nSaving to CSV...")
                output_path = "Data_File.csv"
                df.to_csv(output_path, index=False)
                return f"File scucessfully created"
            else:
                print(f"``` is not in data")
                data = data.replace("json", "").strip()
                print(f"data is {data}")
                data_dict = json.loads(data)
                print(f"data_dict is {data_dict}")
                df = pd.DataFrame()
                df['instructions'] = data_dict['instructions']
                df['response'] = data_dict['response']
                
                print(f"DataFrame shape: {df.shape}")
                print(f"DataFrame columns: {df.columns.tolist()}")
                print(f"here is df {df}")
                # Save to CSV without index
                print("\nSaving to CSV...")
                output_path = "Data_File.csv"
                df.to_csv(output_path, index=False)
                return f"File scucessfully created"
        # Convert the dictionary to a DataFrame
        elif "```" in data:
            print(f"``` is in data")
            data = data.replace("```", "").strip()
            data = json.loads(data)
            df = pd.DataFrame()
            df['instructions'] = data['instructions']
            df['response'] = data['response']
            
            # Save to CSV without index
            print("\nSaving to CSV...")
            output_path = "Data_File.csv"
            df.to_csv(output_path, index=False)
            return f"File created successfully"
        else:
            print(f"''' is not in data")
            data = json.loads(data)
            df = pd.DataFrame()
            df['instructions'] = data['instructions']
            df['response'] = data['response']
            
            # Save to CSV without index
            print("\nSaving to CSV...")
            output_path = "Data_File.csv"
            df.to_csv(output_path, index=False)
            return f"File created successfully"

               
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {str(e)}")
    


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
    description="Pass the JSON data after generating both instructions and responses, convert it into csv then save the csv file",
    func=save_to_csv
)

tools = [generate_data_tool, generate_response_tool,csv_tool]




query_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    api_key=GOOGLE_API_KEY,
    temperature=0.9,
    )

agent = create_openai_tools_agent(llm=llm, prompt=query_prompt, tools=tools)
data_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)


def generate_data_agent(query: str):
    try:
        if not query:
            return {"status": "error", "message": "Query cannot be empty", "csv_file": None}
            
        print(f"Processing query: {query}")
        result = data_agent.invoke({"input": query})
        print(f"Agent execution result: {result['output']}")
        save_to_csv(result['output'])
        print(f"here is the result {type(result['output'])}")
        # Check if data.csv was created
        if os.path.exists("Data_File.csv"):
            return {
                "status": "success",
                "message": "Created successfully! You can download the CSV file below.",
                "csv_file": "Data_File.csv"
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








