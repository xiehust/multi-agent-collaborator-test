from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import sys
from utils import get_model_br_client,get_model_ds_client,get_model_litellm_client


model_client = get_model_litellm_client()

SRC_LANG = "English"
TRT_LANG = "Chinese"
COUNTRY = "China"

#define translate_agent
translate_agent = AssistantAgent("translation_agent", 
    model_client=model_client,
    system_message = """
You are an expert linguist, specializing in translation from {source_lang}to {target_lang}.
please provide the  {target_lang} translation for this text. 
Do not provide any explanations or text apart from the translation.
""".format(source_lang=SRC_LANG,target_lang=TRT_LANG)
)

#define review_agent
review_agent = AssistantAgent("review_agent", 
model_client=model_client,
system_message = """
You are an expert linguist, specializing in translation from {source_lang}to {target_lang}.
You will be provided with a source text and its translation and your goal is to improve the translation.

Your task is to carefully read a source text and a translation from {source_lang}to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \

The final style and tone of the translation should match the style of {source_lang} colloquially spoken in {country}.

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {{target_lang}}).\n\
Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else.
""".format(source_lang=SRC_LANG,target_lang=TRT_LANG,country = COUNTRY)
)

translate_agent_2 = AssistantAgent("chief_translation_agent", 
model_client=model_client,
system_message = """
You are an expert linguist, specializing in translation from {source_lang}to {target_lang}.
Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into account a list of expert suggestions and constructive criticisms.
You will be provided with the source text, the initial translation, and the expert linguist suggestions,

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.
Output only the new translation and nothing else.
""".format(source_lang=SRC_LANG,target_lang=TRT_LANG,country = COUNTRY)
)


user_proxy = UserProxyAgent("user_proxy", input_func=input)  # Use input() to get user input from console.

if __name__ == "__main__":


    termination = TextMentionTermination("APPROVE")

    team = RoundRobinGroupChat([translate_agent, 
                                review_agent,
                                translate_agent_2,
                                user_proxy
                                ], 
                                max_turns=3,
                               termination_condition=termination)

    user_input= "Implementing RAG requires organizations to perform several cumbersome steps to convert data into embeddings (vectors), store the embeddings in a specialized vector database, and build custom integrations into the database to search and retrieve text relevant to the user’s query. This can be time-consuming and inefficient."
    stream = team.run_stream(task=user_input)
    asyncio.run(Console(stream,output_stats=True))

    # print("Welcome to the interactive Multi-Agent system. Type 'quit' to exit.")
    # while True:
    #     # Get user input
    #     user_input = input("\nYou: ").strip()
    #     if user_input.lower() == 'quit':
    #         print("Exiting the program. Goodbye!")
    #         sys.exit()
    #     # Run the conversation and stream to the console.
    #     stream = team.run_stream(task=user_input)
    #     asyncio.run(Console(stream))