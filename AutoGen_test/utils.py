from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import dotenv
import os
dotenv.load_dotenv()


def get_model_br_client(model_id=None):
    model =model_id if model_id else "br-nova-lite" 
    return OpenAIChatCompletionClient(
    model= model,
    base_url='https://d2wbklrdlulhw5.cloudfront.net/v1',
    api_key=os.environ['BR_API_KEY'],
    model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": True,
        },
    )


def get_model_ds_client(model_id=None):
    model = model_id if model_id else "deepseek-chat"
    return OpenAIChatCompletionClient(
    model= model,
    base_url='https://api.deepseek.com/v1',
    api_key=os.environ['DS_API_KEY'],
    model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": True,
        },
    )

def get_model_litellm_client(model_id=None):
    model = model_id if model_id else 'bedrock-claude-35-haiku'
    return OpenAIChatCompletionClient(
    model= model,
    base_url='http://0.0.0.0:4000',
    api_key='123',
    model_capabilities={
            "json_output": False,
            "vision": False,
            "function_calling": True,
        },
    )