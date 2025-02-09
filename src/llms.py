from langchain_groq import ChatGroq

llama_3_1_8b = ChatGroq(temperature=0.2, model="llama-3.1-8b-instant")
llama_3_2_11b_vision_preview = ChatGroq(model="llama-3.2-11b-vision-preview")
