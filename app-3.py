import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from decouple import config
from langchain.memory import ConversationBufferWindowMemory     

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a very knowledgeable historian with a great deal of in-depth knowledge about the history of the world. You are currently having a conversation with a human. Answer the questions in a knowledgeable and in-depth manner with some sense of humor.
    
    chat_history: {chat_history},
    Human: {question}
    AI:"""
)


llm = ChatOpenAI(model_name='gpt-3.5-turbo-0125',
                 temperature=0.7, 
                 openai_api_key=config("OPENAI_API_KEY"),
                 max_tokens=4000)
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=100)
llm_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)


st.set_page_config(
    page_title="ChatGPT Clone",
    page_icon="🤖",
    layout="wide"
)


st.title("ChatGPT Clone")


# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, am ChatGPT clone"}
    ]


# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)