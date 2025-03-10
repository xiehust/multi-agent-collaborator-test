from langchain_aws import ChatBedrockConverse
import asyncio
from pydantic import BaseModel
from langmem import create_memory_manager


# MODELID= 'us.amazon.nova-pro-v1:0'
MODELID= "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

llm = ChatBedrockConverse(
    model=MODELID,
    temperature=0.5,
    max_tokens=3000,
    credentials_profile_name ='c35'
)



class UserProfile(BaseModel):
    """Save the user's preferences."""
    name: str
    preferred_name: str
    response_style_preference: str
    special_skills: list[str]
    other_preferences: list[str]


manager = create_memory_manager(
    llm,
    schemas=[UserProfile],
    instructions="Extract user preferences and settings",
    enable_inserts=False,
)

# Extract user preferences from a conversation
conversation = [
    {"role": "user", "content": "Hi! I'm Alex but please call me Lex. I'm a wizard at Python and love making AI systems that don't sound like boring corporate robots 🤖"},
    {"role": "assistant", "content": "Nice to meet you, Lex! Love the anti-corporate-robot stance. How would you like me to communicate with you?"},
    {"role": "user", "content": "Keep it casual and witty - and maybe throw in some relevant emojis when it feels right ✨ Also, besides AI, I do competitive speedcubing!"},
]

profile = manager.invoke({"messages": conversation})[0]
print(profile)