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
import boto3
import dotenv
import os
dotenv.load_dotenv()

MODELID= 'us.amazon.nova-pro-v1:0'
# MODELID= "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
custom_client = boto3.client('bedrock-runtime',
                             aws_access_key_id=os.environ['ACCESS_KEY_ID'],
                             aws_secret_access_key=os.environ['SECRET_ACCESS_KEY'],
                             config = boto3.session.Config(
                                 read_timeout=120,
                                 connect_timeout=120
                             ))
# br_agent_client = boto3.client('bedrock-agent-runtime', region_name='us-east-1' )

class BedrockLLMAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        # handle response streaming here
        print(token, end='', flush=True)

llm_config= dict(
    model_id=MODELID,
    region='us-east-1',
    inference_config={
        'maxTokens': 3000,
        'temperature': 0.7,
        'topP': 0.9
    },
    client=custom_client
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

# Add individual agents
orchestrator.add_agent(BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Health Assistant",
    description="Focuses on health and medical topics such as general wellness, nutrition, diseases, treatments, mental health, fitness, healthcare systems, and medical terminology or concepts.",
    streaming=True,
    callbacks=BedrockLLMAgentCallbacks(),
    **llm_config,
)))


bedrock_agent =  AmazonBedrockAgent(AmazonBedrockAgentOptions(
        name="Hotel Service",
        description="customer service for booking hotel",
        agent_id="K9MBIIVF72",
        agent_alias_id="GATM33U2WD",
        streaming=True,
        enableTrace=False,
    ))

llm_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
            name="Complaint Agent",
            description="负责处理用户投诉，安抚用户",
            callbacks=BedrockLLMAgentCallbacks(),
            streaming=True,
            **llm_config
        ))

# Create and configure SupervisorAgent as sub agent for another SupervisorAgent
supervisor_agent_1 = SupervisorAgent(SupervisorAgentOptions(
    lead_agent=BedrockLLMAgent(BedrockLLMAgentOptions(
        name="social-media-campaign-manager",
        description="a strategic campaign manager who orchestrates social media campaigns from concept to execution.",
        **llm_config,
    )),
    team=[
        BedrockLLMAgent(BedrockLLMAgentOptions(
            name="content-strategist",
            description="a social media content strategist with expertise in converting business goals into engaging social posts. Your task is to generate creative, on-brand content ideas that align with specified campaign goals and target audience. Each suggestion should include a topic, content type (image/video/text/poll), specific copy, and relevant hashtags. Focus on variety, authenticity, and ensuring each post serves a strategic purpose.",
             **llm_config
        )),
        BedrockLLMAgent(BedrockLLMAgentOptions(
            name="engagement-predictor",
            description="You are a social media analytics expert who predicts post performance and optimal timing. For each content idea, analyze potential reach and engagement based on content type, industry benchmarks, and audience behavior patterns. Your task is to estimate reach, engagement rate, and determine the best posting time (day/hour). Support each prediction with data-driven reasoning and industry-specific insights. Focus on actionable metrics that will maximize campaign impact.",
             **llm_config
        ))
    ]
))

# 测试多个supervisor agent级联
supervisor_agent_2 = SupervisorAgent(SupervisorAgentOptions(
    lead_agent=BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Support Team Lead",
        description="处理酒店客服咨询，预定，投诉等，同时也能做营销助手",
        **llm_config
    )),
    team=[
        bedrock_agent,
        llm_agent,
        supervisor_agent_1
    ]
))

orchestrator.add_agent(supervisor_agent_2)


async def handle_request(_orchestrator: MultiAgentOrchestrator, _user_input: str, _user_id: str, _session_id: str):
    response: AgentResponse = await _orchestrator.route_request(_user_input, _user_id, _session_id)
    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    if response.metadata.agent_name == 'No Agent':
        print('Response:', response)
    elif response.streaming:
        print('Response:', response.output.content[0]['text'])
    else:
        print('Response:', response.output.content[0]['text'])

async def simple_handle_request(agent, _user_input: str, _user_id: str, _session_id: str):
    response: ConversationMessage = await agent.process_request(_user_input, _user_id, _session_id,[])
    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.content}")


if __name__ == "__main__":
    USER_ID = "user123"
    SESSION_ID = str(uuid.uuid4())
    print("Welcome to the interactive Multi-Agent system. Type 'quit' to exit.")
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()
        # Run the async function
        asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))
        # asyncio.run(simple_handle_request(bedrock_agent, user_input, USER_ID, SESSION_ID))


