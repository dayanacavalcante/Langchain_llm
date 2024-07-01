from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vectordb import url_to_retriver
from chatgpt import llm

prompt = ChatPromptTemplate.from_template("""Answer a question based on context alone:
{context}
Question: {input}                            
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriver = url_to_retriver('https://www.euronews.com/my-europe/2024/07/01/french-election-results-winners-and-losers-in-paris')
retriver_chain = create_retrieval_chain(retriver, document_chain)
response = retriver_chain.invoke({"input":"By what margin did the National Rally party win the legislative elections on Sunday?"})
print(response['answer'])