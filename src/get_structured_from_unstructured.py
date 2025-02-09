from unstructured.partition.pdf import partition_pdf
import os
import pickle
import json

from src.chains import summarize_table_text_chain, summarize_iamges_chain
from dotenv import load_dotenv, find_dotenv

# Load the API keys from .env
load_dotenv(find_dotenv(), override=True)
# file_path = "E:\\python projects\\unstructured-pdf\\data\documents\\HFSA Guideline for theManagement of Heart Failure.pdf"


def get_pdf_chunks(
    filepath: str = os.getenv("PDF_FILEPATH"),
    elements_filpath: str = os.getenv("PDF_ELEMENTS_FILEPATH"),
):
    chunks = partition_pdf(
        filename=filepath,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir="E:\\python projects\\unstructured-pdf\\data\\documents\\images",
        # image_output_dir_path=output_path,
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=1000,
        combine_text_under_n_chars=200,
        new_after_n_chars=600,
        # extract_images_in_pdf=True,          # deprecated
    )

    # Save the unstructured output (chunks) as a Pickle file
    with open(
        elements_filpath,
        "wb",
    ) as f:
        pickle.dump(chunks, f)
    return chunks


def load_chunks(filepath: str = os.getenv("PDF_ELEMENTS_FILEPATH")):

    if not os.path.exists(filepath):
        print("File does not exist")
        return None
    else:
        # Load the saved chunks output from Pickle file
        with open(filepath, "rb") as f:
            loaded_chunks = pickle.load(f)
        return loaded_chunks


def separate_tables_and_text(chunks):
    # separate tables from texts
    tables = []
    texts = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)

    return {"texts": texts, "tables": tables}


# Get the images from the CompositeElement objects
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_elements = chunk.metadata.orig_elements
            for el in chunk_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def get_table_summaries(tables, filepath: str = os.getenv("TABLE_SUMMARIES_FILEPATH")):

    print("summarizing tables...")
    # Summarize tables
    tables_html = [table.metadata.text_as_html for table in tables]

    from tqdm import tqdm

    table_summaries = []
    for table in tqdm(tables_html, desc="Summarizing Tables"):
        table_summaries.append(summarize_table_text_chain.invoke(table))

    with open(filepath, "w") as f:
        json.dump(
            [
                {"table_html": table.metadata.text_as_html, "table_summary": summary}
                for table, summary in zip(tables, table_summaries)
            ],
            f,
            indent=4,
        )

    return table_summaries


def load_table_summaries(filepath: str = os.getenv("TABLE_SUMMARIES_FILEPATH")):
    if not os.path.exists(filepath):
        print("File does not exist")
        return None
    else:
        with open(
            filepath,
            "r",
        ) as f:
            table_summaries = json.load(f)
        return table_summaries


def get_image_summaries(images, filepath: str = os.getenv("IMAGE_SUMMARIES_FILEPATH")):
    print("summarizing...")
    image_summaries = summarize_iamges_chain.batch(images)
    with open(
        filepath,
        "w",
    ) as f:
        json.dump(
            [
                {"image_bas64": img, "image_summary": summary}
                for img, summary in zip(images, image_summaries)
            ],
            f,
            indent=4,
        )

    return image_summaries


def load_image_summaries(filepath: str = os.getenv("IMAGE_SUMMARIES_FILEPATH")):
    if not os.path.exists(filepath):
        print("File does not exist")
        return None
    else:
        with open(
            filepath,
            "r",
        ) as f:
            image_summaries = json.load(f)
        return image_summaries
