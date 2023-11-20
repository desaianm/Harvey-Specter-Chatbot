from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from apikey import key
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import streamlit as st
import os
os.environ['OPENAI_API_KEY'] = key

llm = OpenAI(temperature=0.6)


# 1.  Vectorize the response csv data
loader = CSVLoader(file_path="harv_dataset.csv")
documents  =loader.load()
print(len(documents))

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents,embeddings)


# 2. Do Similarilty Search

def retrieve_info(query):
    similar_response = db.similarity_search(query,k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    print(page_contents_array)
    return page_contents_array
# 3. Setup llmchain and prompts
llm = OpenAI(temperature=0)

template = """
You should act like a person. A person who thinks like the response i gave you when he is asked something.

1/ Response should be very similar or even identical to the past responses, 
in terms of length, ton of voice, logical arguments and other details

2/ If the responses are irrelevant, then try to mimic the style of the response to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of responses of how the person respond to questions:
{best_responses}

Please write the best response that person gives:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_responses"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retreival Augmented Generation

def generate_response(message):
    best_responses = retrieve_info(message)
    response = chain.run(message=message, best_responses=best_responses)
    return response

# 5. Building an App with Streamlit
def main():
    st.set_page_config(
        page_title="Ask Harvey's Advice ", page_icon=":person:")

    st.header("Harvey Specter Chatbot")
    message = st.text_area("your  message")

    if message:
        st.write("Generating Harvey's response ...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()

