import os
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool


from langchain_community.vectorstores import Weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import weaviate
from langchain.globals import set_llm_cache
from langchain.cache import RedisCache
import redis
from operator import itemgetter


from langchain.prompts import StringPromptTemplate
import re
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,)
from typing import Union
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.schema.output_parser import StrOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_groq.chat_models import ChatGroq


TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
REDIS_URL = os.getenv('REDIS_URL')
REDIS_TTL = os.getenv('REDIS_TTL')
WEAVIATE_CLIENT_URL = os.getenv('WEAVIATE_CLIENT_URL')
WEAVIATE_COLLECTION_NAME = os.getenv('WEAVIATE_COLLECTION_NAME')
WEAVIATE_COLLECTION_PROPERTY = os.getenv('WEAVIATE_COLLECTION_PROPERTY')
LLM = os.getenv('LLM')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')


search = TavilySearchResults(max_results=1,
                             description=
        "A search engine"
        "Useful for when you need to answer questions about current events. use this tool to search on questions which you don't have answer."
        "Input should be a search query."
    )


# ------------------------- Message History ---------------------

# message_history = RedisChatMessageHistory(
#     url=REDIS_URL, session_id= itemgetter('session_id'), ttl=REDIS_TTL
# )

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    # chat_memory=message_history,
    return_messages=True,
    output_key="output"
)

# ------------------------- Redis cache ---------------------

redis_client = redis.Redis.from_url(REDIS_URL)
set_llm_cache(RedisCache(redis_client))


# ------------------------- Weaviate Client  ---------------------
client = weaviate.Client(
  url=WEAVIATE_CLIENT_URL,
)

vectorstore = Weaviate(client, 
                       WEAVIATE_COLLECTION_NAME, 
                       WEAVIATE_COLLECTION_PROPERTY)

retriever = vectorstore.as_retriever()

# ------------------------- TOOL - Retriever ---------------------

retriever_tool = create_retriever_tool(
    retriever,
    "rdj_search",
    "Search any information only about the actor robert downey jumior or it's related questions. For any questions related to RDJ, you must use this tool!",
)


# -------------------- General Conversation Tool -----------

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You're an Friendly AI assistant, your name is Claro, you can make normal conversations in a friendly manner, below provided the chat history of the AI and Human make use of it for context reference. if the question is standalone then provide Answer to the question on your own. make sure it sounds like human and official assistant:
            \n"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
    
_model = ChatGroq(api_key=GROQ_API_KEY,
                        model=LLM
                        )



def general_conv(query: str):
    """General conversation, Use this tool to make any general conversation with the user."""
    

    chain5 = _prompt | _model | StrOutputParser()
    
    result = chain5.invoke({"input": query, 'chat_history': memory.buffer_as_messages})
    
    
    return result


conversation_tool = StructuredTool.from_function(
    func=general_conv,
    name="gen_conv",
    description="useful for when you need to answer general question or conversations General conversation, Use this tool to make any general conversation or greeting with the user. and provide the actual user input to this tool ",
    return_direct=True
)


tools = [search, retriever_tool, conversation_tool]


template = """You are an AI assistant, Answer the following question as best you can. if the user is making a general conversation/greeting or normal conversation then use the 'gen_conv' tool.  

You have access to the following tools. make sure to use the format, if you don't know the answer then just say I don't know as Final answer. if anything went wrong just inform the user like try again or try after sometime:

if the question is any related or referenced the chat history entities make use of that to obtain the context to answer the the given question effectively. if the given question is a followup question, then find the context and then select the appropriate tool for it. don't use chat history if the question is standalone and make sure to answer the given question alone. if no chat history is provided then continue with the question alone. 

chat_history: \n {chat_history}
End of chat history.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times if needed if you know the answer on your own then skip the Action and Action Input )
Thought: I now know the final answer
Final Answer: the final answer to the original input question. if you know the Final answer in the beginning then give the Final Answer there is no need to give Thought or Observation. 

Begin! and Strict to the Format. always put 'Final Answer' at the beginning of the final answer. and use the tool name as same as given.

Question: {input}
Thought:{agent_scratchpad}"""




class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: list

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools_getter=tools,
    input_variables=["input", "intermediate_steps", "chat_history"],)



class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        
        action = match.group(1).strip()
        action = action.replace('\\', '') if '\\' in action else action
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )

output_parser = CustomOutputParser()


# ------------------- META CHAIN & AGENT(SINGLE ACTION AGENT) INITIALIZATION -------------------------

model =  ChatGroq(api_key=GROQ_API_KEY, 
                  model=LLM)

llm_chain = LLMChain(llm=model, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

# ------------------- DEBUG & VERBOSE -------------------------


# from langchain.globals import set_debug

# set_debug(True)
# from langchain.globals import set_verbose

# set_verbose(True)


# ------------------- AGENT INPUT SCHEMA DEFINITION & AGENT EXECUTOR INITIALIZATION -------------------------


class AgentInput(BaseModel):
    input: str
    # session_id : str
    

agent = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    # verbose=True, 
    handle_parsing_errors=True,
    memory=memory, 
    return_only_outputs=True

)

agent_executor = agent.with_types(
    input_type=AgentInput) | itemgetter('output')
