import base64
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate


def display_base64_image(base64_code):
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    # Display the image
    display(Image(data=image_data))


from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage

# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from base64 import b64decode


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    print("parse_docs")
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    print(f"docs: ")
    print(f"images {len(b64)}: {b64}")
    print(f"texts: {text}")
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    print("build_prompt")
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )
            break
    print(f"prompt: ")
    print(f"{prompt_content}")
    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )
