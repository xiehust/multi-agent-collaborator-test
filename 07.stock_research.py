import uuid
import asyncio
from typing import Optional, List, Dict, Any
import json
import sys
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.agents import (BedrockLLMAgent,SupervisorAgent,AmazonBedrockAgent,
 BedrockLLMAgentOptions,
 SupervisorAgentOptions,
 AmazonBedrockAgentOptions,
 AgentResponse,
 AgentCallbacks)
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.classifiers import BedrockClassifier, BedrockClassifierOptions
from multi_agent_orchestrator.utils import AgentTool,AgentTools
import boto3
import dotenv
import os
import time
dotenv.load_dotenv()

# MODELID= 'us.amazon.nova-pro-v1:0'
MODELID= "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
custom_client = boto3.client('bedrock-runtime',
                             aws_access_key_id=os.environ['ACCESS_KEY_ID'],
                             aws_secret_access_key=os.environ['SECRET_ACCESS_KEY'],
                             config = boto3.session.Config(
                                 read_timeout=120,
                                 connect_timeout=120
                             ))

class BedrockLLMAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        # handle response streaming here
        print(token, end='', flush=True)

llm_config= dict(
    model_id=MODELID,
    region='us-west-2',
    inference_config={
        'maxTokens': 3000,
        'temperature': 0.2,
    },
    # client=custom_client
)

#add mock tool
def get_stock_data(symbol: str) -> Dict[str, Any]:
    """Get stock market data for a given symbol
    :param symbol 
    """
    ret = {"price": 180.25, "volume": 1000000, "pe_ratio": 65.4, "market_cap": "700B"}
    return json.dumps(ret)

def get_news(query: str) -> List[Dict[str, str]]:
    """Get recent news articles about a company
    :params query
    """
    ret = [
        {
            "title": "Tesla Expands Cybertruck Production",
            "date": "2024-03-20",
            "summary": "Tesla ramps up Cybertruck manufacturing capacity at Gigafactory Texas, aiming to meet strong demand.",
        },
        {
            "title": "Tesla FSD Beta Shows Promise",
            "date": "2024-03-19",
            "summary": "Latest Full Self-Driving beta demonstrates significant improvements in urban navigation and safety features.",
        },
        {
            "title": "Model Y Dominates Global EV Sales",
            "date": "2024-03-18",
            "summary": "Tesla's Model Y becomes best-selling electric vehicle worldwide, capturing significant market share.",
        },
    ]
    return json.dumps(ret)

get_stock_data_tool = AgentTool(
    name="get_stock_data",
    func=get_stock_data,
)

get_news_tool = AgentTool(
    name="get_news",
    func=get_news,
)




#Create an Orchestrator:
orchestrator = MultiAgentOrchestrator(
    classifier=BedrockClassifier(BedrockClassifierOptions(**llm_config)), #The default classifier is Claude, here I create a custom classifier  
    options=OrchestratorConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=False,
        NO_SELECTED_AGENT_MESSAGE='对不起，我不太懂你的意思',
        MAX_MESSAGE_PAIRS_PER_AGENT=10
))



# Create and configure SupervisorAgent as sub agent for another SupervisorAgent
planner = SupervisorAgent(SupervisorAgentOptions(
    name =  "SupervisorAgent",
    description =  "You are a supervisor agent that manages the team of agents for deep research",
    lead_agent=BedrockLLMAgent(BedrockLLMAgentOptions(
        name="planner-manager",
        description="a research planning coordinator.",
         custom_system_prompt={"template":
           """You are a research planning coordinator.
    Coordinate market research by delegating to specialized agents:
    - Financial Analyst: For stock data analysis
    - News Analyst: For news gathering and analysis
    - Writer: For compiling final report
    Always send your plan first, then handoff to appropriate agent.
    Always handoff to a single agent at a time.
    Use TERMINATE when research is complete."""
                                  },
        **llm_config,
        streaming= True,
        callbacks=BedrockLLMAgentCallbacks(),
    )),
    team=[
        BedrockLLMAgent(BedrockLLMAgentOptions(
            name="financial_analyst",
            description="For stock data analysis",
            custom_system_prompt={"template":
            """You are a financial analyst.
    Analyze stock market data using the get_stock_data tool.
    Provide insights on financial metrics.
    Always handoff back to planner when analysis is complete."""
                                  },
             **llm_config,
            streaming= True,
            callbacks=BedrockLLMAgentCallbacks(),
            tool_config={
                'tool': AgentTools([get_stock_data_tool]),
                'toolMaxRecursions': 5,
            },
        )),
        BedrockLLMAgent(BedrockLLMAgentOptions(
            name="news_analyst",
            description="a news analyst.",
            custom_system_prompt={"template":
            """You are a news analyst.
    Gather and analyze relevant news using the get_news tool.
    Summarize key market insights from news.
    Always handoff back to planner when analysis is complete."""
                                  },
             **llm_config,
             streaming= True,
             callbacks=BedrockLLMAgentCallbacks(),
            tool_config={
                'tool': AgentTools([get_news_tool]),
                'toolMaxRecursions': 5,
            },
        )),
        BedrockLLMAgent(BedrockLLMAgentOptions(
            name="writer",
            description="financial report writer.",
            custom_system_prompt={"template":
            """You are a financial report writer.
    Compile research findings into clear, concise reports.
    Always handoff back to planner when writing is complete."""
                                  },
             **llm_config,
            streaming= True,
            callbacks=BedrockLLMAgentCallbacks(),
        ))
    ]
))


orchestrator.add_agent(planner)


async def handle_request(_orchestrator: MultiAgentOrchestrator, _user_input: str, _user_id: str, _session_id: str):
    response: AgentResponse = await _orchestrator.route_request(_user_input, _user_id, _session_id)
    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    if response.metadata.agent_name == 'No Agent':
        print('Response:', response)
    elif response.streaming:
        print('Response:', response.output.content[0]['text'])
        print(response.metadata)
    else:
        print('Response:', response.output.content[0]['text'])
        print(response.metadata)

async def simple_handle_request(agent, _user_input: str, _user_id: str, _session_id: str):
    t1 = time.time()
    response: ConversationMessage = await agent.process_request(_user_input, _user_id, _session_id,[])
    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.content}")
    print(f"Duration:{time.time()-t1}")


if __name__ == "__main__":
    USER_ID = "user123"
    SESSION_ID = str(uuid.uuid4())
    task = "Conduct market research for TSLA stock"
    # asyncio.run(handle_request(orchestrator, task, USER_ID, SESSION_ID))
    asyncio.run(simple_handle_request(planner, task, USER_ID, SESSION_ID))



