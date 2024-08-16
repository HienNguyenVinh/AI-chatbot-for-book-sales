import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import openai
from Embeddings import OpenAIEmbedding
from Rag import RAG
from Reflection import Reflection
from SemanticRouter import Route, SemanticRouter, product_samples, chat_samples

load_dotenv()


MONGODB_URL = os.getenv('MONGODB_URL')
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION')
EMBEDDING_NAME = os.getenv('EMBEDDING_NAME')
OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
OPEN_AI_EMBEDDING_MODEL = os.getenv('OPEN_AI_EMBEDDING_MODEL')
OPEN_AI_ORG_ID = os.getenv('OPEN_AI_ORG_ID')
OPEN_AI_BASE_URL = os.getenv('OPEN_AI_BASE_URL')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


OpenAIEmbedding(OPEN_AI_KEY)

PRODUCT_ROUTE_NAME = 'products'
CHITCHAT_ROUTE_NAME = 'chitchat'

openAIEmbeding = OpenAIEmbedding(apiKey=OPEN_AI_KEY, dimensions=1024, name=OPEN_AI_EMBEDDING_MODEL)
productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=product_samples)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chat_samples)
semanticRouter = SemanticRouter(openAIEmbeding, routes=[productRoute, chitchatRoute])

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel('gemini-1.5-pro')

gpt = openai.OpenAI(api_key=OPEN_AI_KEY)
reflection = Reflection(llm=gpt)

rag = RAG(mongodbUrl = MONGODB_URL,
          dbName = DB_NAME,
          dbCollection=DB_COLLECTION,
          embeddingName='keepitreal/vietnamese-sbert',
          llm=llm)

st.title("Semantic Router with RAG and LLM Integration")
st.write("Ask me:")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


def process_query(query):
    return query.lower()

data = []
query = st.text_input("Ask me:")
if st.button("Send"):
    query = process_query(query)
    data.append({'role': 'user', 'parts': [{'text': query}]})
    if not query:
        st.error("Please enter a query.")
    else:
        guided_route = semanticRouter.route(query)

        if guided_route == PRODUCT_ROUTE_NAME:
            print("Guide to RAGs")
            reflected_query = reflection(data
                                         )
            source_information = rag.get_full_prompt(reflected_query).replace('<br>', '\n')
            combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng bán sách. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."

            data.append({
                "role": "user",
                "parts": [
                    {
                        "text": combined_information,
                    }
                ]
            })
            response = rag.generate_content(data)

            st.session_state['chat_history'].append({"role": "user", "text": query})
            st.session_state['chat_history'].append({"role": "model", "text": response.text})
        else:
            print("Guide to LLMs")
            response = llm.generate_content(data)
            st.session_state['chat_history'].append({"role": "user", "text": query})
            st.session_state['chat_history'].append({"role": "model", "text": response.text})

st.write("### Chat History:")
for chat in st.session_state['chat_history']:
    if chat['role'] == 'user':
        st.write(f"**You:** {chat['text']}")
    else:
        st.write(f"**Bot:** {chat['text']}")
