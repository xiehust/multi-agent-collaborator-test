from langmem import create_prompt_optimizer
from langchain_aws import ChatBedrockConverse
import asyncio
# MODELID= 'us.amazon.nova-pro-v1:0'
MODELID= "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

llm = ChatBedrockConverse(
    model=MODELID,
    temperature=0.5,
    max_tokens=3000,
    credentials_profile_name ='c35'
)

optimizer_grad = create_prompt_optimizer(llm,kind='gradient')

optimizer_prompt_memory = create_prompt_optimizer(llm,kind='prompt_memory')

optimizer_metaprompt = create_prompt_optimizer(llm,kind='metaprompt')


# Example conversation with feedback
conversation = [
    {"role": "user", "content": "Tell me about the solar system"},
    {"role": "assistant", "content": "The solar system consists of..."},
]
feedback = {"clarity": "needs more structure"}

# Use conversation history to improve the prompt
trajectories = [(conversation, feedback)]

async def run_optimizer(optimizer):
    better_prompt = await optimizer.ainvoke(
        {"trajectories": trajectories, "prompt": "You are an astronomy expert"}
    )
    print(f"{better_prompt}")

if __name__ == "__main__":
    print("metaprompt\n")
    asyncio.run(run_optimizer(optimizer_metaprompt))
