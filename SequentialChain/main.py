from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from dotenv import load_dotenv
load_dotenv()


llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    template="Write a test for the following {language} code:\n{code}",
    input_variables=["language", "task"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key='code'
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key='test'
)

chain = SequentialChain(chains=[code_chain,test_chain],
                        input_variables=["task","language"],
                        output_variables=['test','code'])

result = chain({
    "language": "python",
    "task": "return a list of numbers"
})

print("Generated code:")
print(result['code'])
print()
print('Generated test:')
print(result['test'])