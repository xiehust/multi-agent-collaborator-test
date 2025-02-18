import uuid
import asyncio
from typing import Optional, List, Dict, Any
import json
import sys
from multi_agent_orchestrator.orchestrator import MultiAgentOrchestrator, OrchestratorConfig
from multi_agent_orchestrator.agents import (
    BedrockLLMAgent,
    SupervisorAgent,
    AmazonBedrockAgent,
 BedrockLLMAgentOptions,
 SupervisorAgentOptions,
 AmazonBedrockAgentOptions,
 AgentResponse,
 ChainAgent, 
 ChainAgentOptions,
 ComprehendFilterAgent,
 ComprehendFilterAgentOptions,
 AgentCallbacks)
from multi_agent_orchestrator.types import ConversationMessage, ParticipantRole
from multi_agent_orchestrator.classifiers import BedrockClassifier, BedrockClassifierOptions
import boto3
import dotenv
import os
dotenv.load_dotenv()

MODELID= 'us.amazon.nova-lite-v1:0'
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
        print(token, end='', flush=True)

llm_config= dict(
    model_id=MODELID,
    region='us-east-1',
    inference_config={
        'maxTokens': 3000,
        'temperature': 0.7,
        'topP': 0.9
    },
    # client=custom_client
)


translate_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name='Translation  Agent',
    description='Translation Agent',
    **llm_config
))

translate_agent.set_system_prompt(
template = """
You are an expert linguist, specializing in translation from {{source_lang}}to {{target_lang}}.
please provide the  {{target_lang}} translation for this text. 
Do not provide any explanations or text apart from the translation.
""")

review_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name='Review Agent',
    description='Review Agent reviews the translations',
    **llm_config
))

review_agent.set_system_prompt(
"""
You are an expert linguist, specializing in translation from {{source_lang}}to {{target_lang}}.
You will be provided with a source text and its translation and your goal is to improve the translation.

Your task is to carefully read a source text and a translation from {{source_lang}}to {{target_lang}}, and then give constructive criticism and helpful suggestions to improve the translation. \

The final style and tone of the translation should match the style of {{source_lang}} colloquially spoken in {{country}}.

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {{target_lang}} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {{target_lang}}).\n\
Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else.
"""
)

translate_agent_2 = BedrockLLMAgent(BedrockLLMAgentOptions(
    name='Chief Translation Agent',
    description='Chief Translation Agent for final translation',
    **llm_config
))

translate_agent_2.set_system_prompt(
template = """
You are an expert linguist, specializing in translation from {{source_lang}}to {{target_lang}}.
Your task is to carefully read, then edit, a translation from {{source_lang}} to {{target_lang}}, taking into account a list of expert suggestions and constructive criticisms.
You will be provided with the source text, the initial translation, and the expert linguist suggestions,
## The source text as follows:
{{user_input}}

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {{target_lang}} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.
Output only the new translation and nothing else.
""")

# filter_agent = ComprehendFilterAgent(ComprehendFilterAgentOptions(
#     name='ContentModerator',
#     description='Analyzes and filters content using Amazon Comprehend',
# ))


chain_agent = ChainAgent(ChainAgentOptions(
    name='TranslationChainAgent',
    description='A simple translation chain of multiple agents',
    agents=[translate_agent, review_agent,translate_agent_2]
))

orchestrator = MultiAgentOrchestrator(
    classifier=BedrockClassifier(BedrockClassifierOptions(**llm_config)), #The default classifier is Claude, here I create a custom classifier  
    default_agent = chain_agent,
    options=OrchestratorConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        NO_SELECTED_AGENT_MESSAGE='对不起，我不太懂你的意思',
        MAX_MESSAGE_PAIRS_PER_AGENT=10
))

orchestrator.add_agent(chain_agent)

async def handle_request(_orchestrator: MultiAgentOrchestrator, _user_input: str, _user_id: str, _session_id: str):
    response: AgentResponse = await _orchestrator.route_request(_user_input, _user_id, _session_id)
    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.metadata.agent_name}")
    if response.metadata.agent_name == 'No Agent':
        print('Response:', response)
    else:
        print('Response:', response.output.content[0]['text'])

async def simple_handle_request(agent, _user_input: str, _user_id: str, _session_id: str):
    response: ConversationMessage = await agent.process_request(_user_input, _user_id, _session_id,[])
    # Print metadata
    print("\nMetadata:")
    print(f"Selected Agent: {response.content}")


if __name__ == "__main__":
    translate_agent.set_system_prompt(variables = {"source_lang":"English","target_lang":"Chinese" })
    review_agent.set_system_prompt(variables = {"source_lang":"English",
                                                 "target_lang":"Chinese",
                                                   "country":"China" })
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
        # asyncio.run(simple_handle_request(chain_agent, user_input, USER_ID, SESSION_ID))
        translate_agent_2.set_system_prompt(variables = {"source_lang":"English", "target_lang":"Chinese", "user_input":user_input })
        asyncio.run(handle_request(orchestrator, user_input, USER_ID, SESSION_ID))