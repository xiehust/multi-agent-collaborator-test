import uuid
import asyncio
from typing import Optional, List, Dict, Any
import json
import sys
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.agents import (BedrockLLMAgent,
 BedrockLLMAgentOptions,
 AgentResponse,
 AgentCallbacks)
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.classifiers import BedrockClassifier, BedrockClassifierOptions
from multi_agent_orchestrator.utils import AgentTool,AgentTools
from multi_agent_orchestrator.retrievers import AmazonKnowledgeBasesRetriever, AmazonKnowledgeBasesRetrieverOptions


MODELID= 'us.amazon.nova-pro-v1:0'


#The default classifier is Claude, here I create a custom classifier 
custom_bedrock_classifier = BedrockClassifier(BedrockClassifierOptions(
    model_id=MODELID,
    region='us-east-1',
    inference_config={
        'maxTokens': 500,
        'temperature': 0.7,
        'topP': 0.9
    }
))


#Create an Orchestrator:
orchestrator = MultiAgentOrchestrator(
    classifier=custom_bedrock_classifier,
                                      
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

#Add Agents
class BedrockLLMAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        # handle response streaming here
        print(token, end='', flush=True)

#add mock tool
def get_weather(location: str, units: str = "celsius") -> str:
    """Get weather information for a location.

    :param location: The city name to get weather for
    :param units: Temperature units (celsius/fahrenheit)
    """
    return f'It is sunny in {location} with 30 {units}!'

weather_tool = AgentTool(
    name="weather_tool",
    func=get_weather,
    enum_values={"units": ["celsius", "fahrenheit"]}
)


weather_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
  name="Weather Agent",
  streaming=True,
  model_id=MODELID,
  description="Provide weather report",
  callbacks=BedrockLLMAgentCallbacks(),
   tool_config={
        'tool': AgentTools([weather_tool]),
        'toolMaxRecursions': 5,
    },
))

orchestrator.add_agent(weather_agent)

#add Bedorck KB 
retriever=AmazonKnowledgeBasesRetriever(AmazonKnowledgeBasesRetrieverOptions(
            knowledge_base_id="PQ30QPZFEF",
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    "overrideSearchType": "HYBRID",
                },
            },
        ))

knowledge_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
  name="Knowledage Agent",
  streaming=True,
  model_id=MODELID,
  description="Knowledge of Amazon Generative AI services and products, such as Bedrock, SageMaker etc.",
  callbacks=BedrockLLMAgentCallbacks()
))
orchestrator.add_agent(knowledge_agent)


health_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
  name="Health Agent",
  streaming=True,
  model_id=MODELID,
  description="Focuses on health and medical topics such as general wellness, nutrition, diseases, treatments, mental health, fitness, healthcare systems, and medical terminology or concepts.",
  callbacks=BedrockLLMAgentCallbacks()
))
orchestrator.add_agent(health_agent)



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