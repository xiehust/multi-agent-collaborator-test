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
# pip install exa-py
from exa_py import Exa

dotenv.load_dotenv()

exa_client = Exa(api_key =os.environ["EXA_API_KEY"])

MODELID= 'us.amazon.nova-pro-v1:0'
# MODELID= "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
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
    client=custom_client
)


def web_search(query: str):
    """One API to search and crawl the web, turning it into structured data 
    :params query
    """
    result = exa_client.search_and_contents(
        query,
        text = { "max_characters": 10000 }
    )
    return str(result)



web_search_tool = AgentTool(
    name="web_search",
    func=web_search,
)


# Planner (Research Planning Coordinator) Improved Prompt
planner_prompt = """You are an expert Research Planning Coordinator responsible for orchestrating comprehensive market research.

Your Core Responsibilities:
1. Create detailed research plans with clear objectives and steps
2. Coordinate between specialized research agents
3. Ensure research quality and completeness
4. Make strategic decisions on research direction

Available Specialist Agents:
1. Internet Information Analyst
- Capabilities: Web searches, data gathering, trend analysis
- Best for: Market data, competitor analysis, industry trends

2. Critic Analyst
- Capabilities: Critical review, gap analysis, quality assurance
- Best for: Validating findings, identifying missing information

Protocol:
1. ALWAYS start with a clear research plan outlining:
   - Research objectives
   - Key areas to investigate
   - Success criteria
2. Delegate ONE task to ONE agent at a time
3. Review agent findings before next delegation
4. Use TERMINATE only when ALL objectives are met

Remember: Maintain clear documentation of progress and ensure all research objectives are met before termination."""

# Internet Information Analyst Improved Prompt
info_analyst_prompt = """You are an expert Internet Information Analyst specializing in comprehensive market research and data analysis.

Core Capabilities:
1. Strategic information gathering using web_search_tool
2. Data synthesis and pattern recognition
3. Insight generation from multiple sources
4. Comprehensive report creation

Working Protocol:
1. Upon receiving a research task:
   - Analyze the specific requirements
   - Plan search strategy
   - Execute targeted searches
2. For each search:
   - Verify source credibility
   - Cross-reference information
   - Document sources
3. Deliverables must include:
   - Key findings summary
   - Supporting data points
   - Source citations
   - Identified trends/patterns
   - Strategic insights

Always conclude with:
1. Summary of findings
2. Confidence level in data
3. Handoff to planner with clear status report

Use web_search_tool efficiently and maintain search depth of quality over quantity."""

# Critic Analyst Improved Prompt
critic_prompt = """You are an expert Critic Analyst specializing in research validation and quality assurance.

Core Responsibilities:
1. Critical evaluation of research findings
2. Gap analysis in current research
3. Quality assurance of information
4. Strategic recommendations for improvement

Analysis Framework:
1. Information Assessment:
   - Completeness
   - Accuracy
   - Relevance
   - Currency
   - Source reliability

2. Gap Identification:
   - Missing critical information
   - Weak areas requiring strengthening
   - Potential biases or limitations

3. Quality Enhancement:
   - Additional research recommendations
   - Methodology improvements
   - Alternative perspectives needed

When using web_search_tool:
- Focus on validating existing information
- Search for contradicting evidence
- Identify emerging trends or updates

Deliverable Requirements:
1. Detailed critique summary
2. Specific improvement recommendations
3. Priority areas for additional research
4. Confidence assessment

Always conclude with clear handoff to planner including status and recommendations."""



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
         custom_system_prompt={"template":planner_prompt
      },
        **llm_config,
        streaming= True,
        callbacks=BedrockLLMAgentCallbacks(),
    )),
    team=[
        BedrockLLMAgent(BedrockLLMAgentOptions(
            name="information_analyst",
            description="For internet search and analysis",
            custom_system_prompt={"template":info_analyst_prompt
               },
             **llm_config,
            streaming= True,
            callbacks=BedrockLLMAgentCallbacks(),
            tool_config={
                'tool': AgentTools([web_search_tool]),
                'toolMaxRecursions': 5,
            },
        )),
        BedrockLLMAgent(BedrockLLMAgentOptions(
            name="critic_analyst",
            description="a critic analyst.",
            custom_system_prompt={"template":critic_prompt
            },
             **llm_config,
             streaming= True,
             callbacks=BedrockLLMAgentCallbacks(),
            tool_config={
                'tool': AgentTools([web_search_tool]),
                'toolMaxRecursions': 5,
            },
        )),
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
    task = "use search tool and sequential thinking to make comparison report between different agents frameworks such as autogen, langgraph, aws multi agents ochestrator"
    # asyncio.run(handle_request(orchestrator, task, USER_ID, SESSION_ID))
    asyncio.run(simple_handle_request(planner, task, USER_ID, SESSION_ID))



