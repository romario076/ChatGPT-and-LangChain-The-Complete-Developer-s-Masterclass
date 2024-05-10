from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv
load_dotenv()

chat = ChatOpenAI(model_name='gpt-3.5-turbo', verbose=True)

#memory = ConversationSummaryMemory(memory_key="messages", return_messages=True,
# chat_memory=FileChatMessageHistory('messages.json'))

memory = ConversationSummaryMemory(memory_key="messages", return_messages=True,
                                  chat_memory=FileChatMessageHistory('messages.json'),
                                  llm=chat)
prompt = ChatPromptTemplate(
    input_variables=["content","messages"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}"),
        MessagesPlaceholder(variable_name='messages')
    ]
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input(">> ")
    print(f"You entered: {content}")

    result = chain({"content": content})
    print(result["text"])
