from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt
summarize_table_text_prompt_text = """
**Role**: **Content Summarizer**

**Task**: Summarize the following table or text chunk concisely, highlighting the key points. 
Provide only the summary and avoid extra commentary, explanations, or introductory phrases.
Nothing extra should be there like "keywords:", "summary:" etc.

**Table or text chunk**: {element}
"""
summarize_table_text_prompt = ChatPromptTemplate.from_template(
    summarize_table_text_prompt_text
)


summarize_images_prompt_template = """Provide a detailed description of the image, focusing on any medical data visualizations. 
If the image includes graphs such as bar plots, scatter plots, or other charts, 
describe them in detail, including axes, labels, and key trends relevant to the medical context."""
messages = [
    (
        "user",
        [
            {"type": "text", "text": summarize_images_prompt_template},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}"},
            },
        ],
    )
]

summarize_images_prompt = ChatPromptTemplate.from_messages(messages)
