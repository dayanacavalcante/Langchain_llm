from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tools import pre_process

load_dotenv()

def legislative_elections(country, year, llm):
    
    prompt = PromptTemplate(
        input_variables=['country', 'year'],
        template = "Which party won the first round of legislative elections in {country} in {year}?"
    )
    elections_chain = LLMChain(llm=llm, prompt=prompt)
    
    response = elections_chain({'country': country, 'year': year})
    
    return response

llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")

if __name__=="__main__":
    pre_process()
    response = legislative_elections('France', 2024, llm)
    print(response['text'])
    