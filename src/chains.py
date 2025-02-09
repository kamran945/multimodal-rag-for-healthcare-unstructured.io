from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage


from src.prompts import summarize_table_text_prompt, summarize_images_prompt
from src.llms import llama_3_1_8b, llama_3_2_11b_vision_preview
from src.utils import parse_docs, build_prompt

summarize_table_text_chain = (
    {"element": lambda x: x}
    | summarize_table_text_prompt
    | llama_3_1_8b
    | StrOutputParser()
)


summarize_iamges_chain = (
    summarize_images_prompt | llama_3_2_11b_vision_preview | StrOutputParser()
)


def get_rag_chain(retriever):
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        # | ChatGroq(model="llama-3.1-8b-instant")
        | llama_3_2_11b_vision_preview
        | StrOutputParser()
    )

    return chain


def get_rag_chain_with_sources(retriever):

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            # | ChatGroq(model="llama-3.1-8b-instant")
            | llama_3_2_11b_vision_preview
            | StrOutputParser()
        )
    )

    return chain_with_sources
