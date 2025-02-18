import asyncio
import uuid
import sys
from multi_agent_orchestrator.agents import BedrockInlineAgent, BedrockInlineAgentOptions
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

action_groups_list = [
    {
        'actionGroupName': 'CodeInterpreterAction',
        'parentActionGroupSignature': 'AMAZON.CodeInterpreter',
        'description':'Use this to write and execute python code to answer questions and other tasks.'
    },
    {
        "actionGroupExecutor": {
            "lambda": "arn:aws:lambda:region:0123456789012:function:my-function-name"
        },
        "actionGroupName": "MyActionGroupName",
        "apiSchema": {
            "s3": {
                "s3BucketName": "bucket-name",
                "s3ObjectKey": "openapi-schema.json"
            }
        },
        "description": "My action group for doing a specific task"
    }
]

knowledge_bases = [
    {
        "knowledgeBaseId": "PQ30QPZFEF",
        "description": 'This is my knowledge base for morgan stanley',
    },
    {
        "knowledgeBaseId": "7L73BZIBHC",
        "description": 'This is my knowledge base for memgpt paper',
    }
]

bedrock_inline_agent = BedrockInlineAgent(BedrockInlineAgentOptions(
    name="Inline Agent Creator for Agents for Amazon Bedrock",
    description="Specalized in creating Agent to solve customer request dynamically. You are provided with a list of Action groups and Knowledge bases which can help you in answering customer request",
    # action_groups_list=action_groups_list,
    client = boto3.client('bedrock-runtime'),
    bedrock_agent_client = boto3.client(
                    'bedrock-agent-runtime',
                    region_name='us-east-1'
                ),
    knowledge_bases=knowledge_bases,
))

async def run_inline_agent(user_input, user_id, session_id):
    response = await bedrock_inline_agent.process_request(user_input, user_id, session_id, [], None)
    return response

if __name__ == "__main__":

    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    print("Welcome to the interactive Multi-Agent system. Type 'quit' to exit.")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Exiting the program. Goodbye!")
            sys.exit()

        # Run the async function
        response = asyncio.run(run_inline_agent(user_input=user_input, user_id=user_id, session_id=session_id))
        print(response.content[0].get('text','No response'))