import argparse
import time
import functools
import yaml
import logging
from enum import Enum, auto
import json
import os
import re
from pathlib import Path
import arxiv
from llama_parse import LlamaParse
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.evaluation import GuidelineEvaluator
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import pymupdf4llm
from scirate import fetch_top_arxiv_paper_ids
from ragflow_sdk import RAGFlow
from ragflow import upload_to_ragflow


logger = logging.getLogger(__name__)


def create_llm(model_name, base_url):
    return Ollama(model=model_name, base_url=base_url, request_timeout=120.0)


def create_embedding_model(model_name, base_url):
    return OllamaEmbedding(model_name=model_name, base_url=base_url)


def create_evaluator(model_name, base_url, guideline):
    evaluator_llm = create_llm(model_name, base_url)
    evaluator = GuidelineEvaluator(
        llm=evaluator_llm,
        guidelines=guideline,
    )
    return evaluator_llm, evaluator


def post_generation(response):
    # for DeepSeek
    if (pos := response.find('</think>')) >= 0:
        return response[pos+len('</think>'):].strip()
    else:
        return response

    # otherwise
    # return response


def complete(llm, prompt):
    return post_generation(str(llm.complete(prompt)))


def call_query_engine(query_engine, query):
    return post_generation(str(query_engine.query(query)))


def download_papers(
    topics,
    num_papers_per_topic,
    dir_path,
    titles_file_path,
    use_scirate=False,
    target_date=None,
    paper_ids=None,
):
    titles = []
    client = arxiv.Client()

    if paper_ids is not None:
        for result in client.results(arxiv.Search(id_list=paper_ids)):
            titles.append(result.title)
            result.download_pdf(dirpath=dir_path)
            time.sleep(1)
    else:
        for topic in topics:
            if use_scirate:
                search = arxiv.Search(
                    id_list=fetch_top_arxiv_paper_ids(
                                topic,
                                target_date,
                                max_result=num_papers_per_topic,
                            ))
            else:
                search = arxiv.Search(
                    query=topic,
                    max_results=num_papers_per_topic,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                )

            for result in client.results(search):
                titles.append(result.title)
                result.download_pdf(dirpath=dir_path)
                time.sleep(1)

    with open(titles_file_path, 'w') as f:
        json.dump(titles, f)


def list_papers(directory):
    return [
        os.path.join(directory, file.name)
        for file in Path(directory).glob('*.pdf')
    ]


def parse_papers(file_paths):
    documents = []
    llama_reader = pymupdf4llm.LlamaMarkdownReader()

    for index, file_path in enumerate(file_paths):
        title = ' '.join(file_path.split('.')[2].split('_'))
        print(
            f"Processing file {index+1}/{len(file_paths)}: {title}"
        )
        document = llama_reader.load_data(file_path)

        for x in document:
            x.metadata['title'] = title

        documents.extend(document)

    return documents


def get_outline(file_path):
    with open(file_path) as f:
        outline = f.read()
    return outline


def create_outline(outline_file_path, titles_file_path):
    template = get_outline(outline_file_path)

    with open(titles_file_path) as f:
        titles = json.load(f)

    outline = []

    latest_papers_section_num = None
    lines = template.splitlines()
    for line in lines:
        outline.append(line)
        section_match = re.match(r'^(\d+)\.\s*(.*)$', line.strip('# '))
        if section_match:
            section_num, section_title = section_match.groups()
            if 'latest papers' in section_title.lower():
                latest_papers_section_num = section_num

                outline.extend(f'{section_num}.{index+1}. {title}'
                               for index, title in enumerate(titles))

    if latest_papers_section_num is None:
        raise ValueError('The outline must include "Latest Papers" section.')

    return '\n'.join(outline)


def extract_title(outline):
    return outline.strip().split('\n')[0].strip('# ').strip()


def generate_query_with_llm(llm, title, section, subsection, instruction):

    prompt = (
        f"Generate a research query for a report on {title}. "
        f"The query should be for the subsection '{subsection}' under the main section '{section}'. "
        f"{instruction} "
        "Simply output the result without any preliminaries.\n\n"
        "result: ")

    response = complete(llm, prompt)
    return str(response).strip()


def classify_query(llm, query):
    prompt = f"""Classify the following query as either "LLM" if it can be answered directly by a large language model with general knowledge, or "INDEX" if it likely required querying an external index or database for specific or up-to-date information.

Query: "{query}"

Consider the following:
1. If the query asks for general knowledge, concepts, or explanations, classify as "LLM".
2. If the query asks for specific facts, recent events, or detailed information that might not be in the LLM's training data, classify as "INDEX".
3. If unsure, err on the side of "INDEX".
4. Simply answer with either “LLM” or “INDEX".

class: """
    response = complete(llm, prompt)
    logger.debug(f"classify query response: {response}")
    classification = str(response).strip().upper()

    if classification not in ["LLM", "INDEX"]:
        classification = "INDEX"

    return classification

def generate_section_content(llm, query_engine, classifier, evaluator, title, current_section, subsection, query_instruction, answer_instruction=None, max_retry=0):

    if answer_instruction is None:
        answer_instruction = 'Please answer this query.'

    for i in range(max(0, max_retry) + 1):
        query = generate_query_with_llm(llm, title, current_section, subsection, query_instruction)

        if 'Latest Papers' in current_section:
            classification = 'INDEX'
        else:
            classification = classify_query(classifier, query)

        logger.debug(f"Query: {query}")
        logger.debug(f"Classification: {classification}")

        if classification == "LLM":
            answer = str(complete(llm, query + ' ' + answer_instruction))
            return query, classification, answer
        else:
            answer = str(call_query_engine(query_engine, query + ' ' + answer_instruction))
            eval_result = evaluator.evaluate(query=query, response=answer)

            logger.debug(f"Answer: {answer}")
            logger.debug(f"Pass: {eval_result.passing}")
            logger.debug(f"Feedback: {eval_result.feedback}")

            if eval_result.passing:
                return query, classification, answer

    return None


def generate_section_contents(llm, index, classifier, evaluator, outline, section_contents_file_path, query_instruction, answer_instruction=None, max_retry=0):
    query_engine = index.as_query_engine(llm=llm)

    lines = outline.strip().split('\n')
    title = extract_title(outline)
    current_section = ""
    section_contents = {}

    for line in lines[1:]:
        if line.startswith('## '):
            current_section = line.strip('# ').strip()
            section_contents[current_section] = {}
        elif re.match(r'^\d+\.\d+\.', line):
            subsection = line.strip()

            section_content = generate_section_content(
                    llm,
                    query_engine,
                    classifier,
                    evaluator,
                    title,
                    current_section,
                    subsection,
                    query_instruction,
                    answer_instruction,
                    max_retry,
            )

            if section_content is None:
                raise Exception(f'Failed to create an appropriate section content for {current_section} > {subsection}')

            query, classification, answer = section_content
            section_contents[current_section][subsection] = {
                    "query": query,
                    "classification": classification,
                    "answer": answer,
            }

    for section in section_contents:
        if any(exclude in section.lower()
               for exclude in ['introduction', 'conclusion']):
            continue

        if not section_contents[section]:
            query = generate_query_with_llm(llm, title, section, "General overview", query_instruction)
            answer = str(complete(llm, query + " Give a short answer."))
            section_contents[section]["General"] = {
                    "query": query,
                    "classification": "LLM",
                    "answer": answer,
            }

    with open(section_contents_file_path, 'w') as f:
        json.dump(section_contents, f)

    return section_contents


def load_section_contents(section_contents_file_path):
    with open(section_contents_file_path) as f:
        section_contents = json.load(f)
    return section_contents


def get_subsections_content(subsections, report):
    for subsection in sorted(subsections.keys(),
                             key=lambda x: int(re.search(r'\d+\.(\d+)', x).group(1))
                             if re.search(r'(\d+\.\d+)', x) else x):
        content = subsections[subsection]['answer']
        subsection_match = re.search(r'(\d+\.\d+)\.\s*(.+)', subsection)
        if subsection_match:
            subsection_num, subsection_title = subsection_match.groups()
            report += f"## {subsection_num} {subsection_title}\n\n{content}\n\n"
        else:
            report += f"## {subsection}\n\n{content}\n\n"
    return report


def format_report(llm, section_contents, outline, introduction_instruction, conclusion_instruction):
    report = ""

    for section, subsections in section_contents.items():
        section_match = re.match(r'^(\d+\.)\s*(.*)$', section)
        if section_match:
            section_num, section_title = section_match.groups()

            if "introduction" in section.lower():
                introduction_num, introduction_title = section_num, section_title
            elif "conclusion" in section.lower():
                conclusion_num, conclusion_title = section_num, section_title
            else:
                combined_content = "\n".join(x['answer'] for x in subsections.values())
                summary_query = f"Provide a short summary for section '{section}':\n\n{combined_content}"
                section_summary = str(complete(llm, summary_query))
                report += f"# {section_num} {section_title}\n\n{section_summary}\n\n"

                report = get_subsections_content(subsections, report)

    introduction_query = f"Create an introduction for the report. {introduction_instruction}\n\nreport:\n\n{report}"
    introduction = str(complete(llm, introduction_query))
    logger.debug(f'Introduction: {introduction}')
    report = f"# {introduction_num} {introduction_title}\n\n{introduction}\n\n{report}"

    conclusion_query = f"Create a conclusion for the report. {conclusion_instruction}\n\nreport:\n\n{report}"
    conclusion = str(complete(llm, conclusion_query))
    logger.debug(f'Conclusion: {conclusion}')
    report += f"# {conclusion_num} {conclusion_title}\n\n{conclusion}"

    title = extract_title(outline)
    report = f"# {title}\n\n{report}"

    return report


def create_index(pdf_dir, index_dir):
    documents = parse_papers(list_papers(pdf_dir))
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=index_dir)

    return index


def load_index(index_dir):
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_dir))
    return index


def generate_report(llm, index, outline, section_contents, result_file_path, introduction_instruction, conclusion_instruction):
    report = format_report(llm, section_contents, outline, introduction_instruction, conclusion_instruction)

    with open(result_file_path, 'w') as f:
        f.write(report)


class Phase(Enum):
    DOWNLOAD = auto()
    INDEX = auto()
    SC = auto()
    REPORT = auto()
    RAGFLOW = auto()


class LazyValue:
    def __init__(self, func):
        self.func = func

    @functools.cached_property
    def value(self):
        return self.func()


def resolve_path(path):
    if os.path.isabs(path):
        return path
    else:
        return os.path.join(os.path.dirname(__file__), path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        action='store_true',
                        help='enable debug messages')
    parser.add_argument('--config',
                        default='config.yml',
                        help='path to config file')
    return parser.parse_args()


def init_logger(log_level):
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logger.addHandler(ch)


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    init_logger(log_level)

    with open(resolve_path(args.config)) as f:
        conf = yaml.safe_load(f)

    phases = [Phase[x] for x in conf['phases']]

    use_scirate = conf.get('scirate', False)
    target_date = conf.get('target_date', None)
    topics = conf.get('topics', None)
    num_papers_per_topic = int(conf.get('num_papers_per_topic', 0))
    paper_ids = conf.get('paper_ids', None)

    outline_file_path = resolve_path(conf['outline_file_path'])
    work_dir = resolve_path(conf['work_dir'])
    max_retry = int(conf['max_retry'])
    llm_model_name = conf['llm_model_name']
    embedding_model_name = conf['embedding_model_name']
    evaluator_model_name = conf['evaluator_model_name']
    ollama_base_url = conf['ollama_base_url']
    evaluator_guideline = conf['evaluator_guideline']
    section_content_query_instruction = conf['section_content_query_instruction']
    section_content_answer_instruction = conf['section_content_answer_instruction']
    introduction_instruction = conf['introduction_instruction']
    conclusion_instruction = conf['conclusion_instruction']
    ragflow_api_key = conf['ragflow_api_key']
    ragflow_base_url = conf['ragflow_base_url']
    ragflow_dataset_id = conf['ragflow_dataset_id']

    ## init
    pdf_dir = os.path.join(work_dir, 'papers')
    titles_file_path = os.path.join(work_dir, 'pdf_titles.json')
    index_dir = os.path.join(work_dir, 'index')
    section_contents_file_path = os.path.join(work_dir, 'section_contents.json')
    result_file_path = os.path.join(work_dir, 'report.md')

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    llm = LazyValue(
        functools.partial(create_llm,
                          model_name=llm_model_name,
                          base_url=ollama_base_url))
    embed_model = LazyValue(
        functools.partial(create_embedding_model,
                          model_name=embedding_model_name,
                          base_url=ollama_base_url))
    evaluator = LazyValue(
        functools.partial(create_evaluator,
                          model_name=evaluator_model_name,
                          base_url=ollama_base_url,
                          guideline=evaluator_guideline))
    outline = LazyValue(
        functools.partial(create_outline,
                          outline_file_path=outline_file_path,
                          titles_file_path=titles_file_path))

    ## download
    phase = Phase.DOWNLOAD
    if phase in phases:
        logger.info(f'phase: {phase.name}')
        download_papers(
            topics,
            num_papers_per_topic,
            pdf_dir,
            titles_file_path,
            use_scirate,
            target_date,
            paper_ids,
        )
        logger.info('done')
    else:
        logger.info(f'phase: {phase.name} (skipped)')

    ## index
    phase = Phase.INDEX
    if phase in phases:
        logger.info(f'phase: {phase.name}')
        Settings.embed_model = embed_model.value
        create_index(pdf_dir, index_dir)
        logger.info('done')
    else:
        logger.info(f'phase: {phase.name} (skipped)')

    index = LazyValue(functools.partial(load_index, index_dir=index_dir))

    ## section contents
    phase = Phase.SC
    if phase in phases:
        logger.info(f'phase: {phase.name}')
        Settings.embed_model = embed_model.value
        generate_section_contents(
            llm.value,
            index.value,
            evaluator.value[0],
            evaluator.value[1],
            outline.value,
            section_contents_file_path,
            section_content_query_instruction,
            section_content_answer_instruction,
            max_retry,
        )
        logger.info('done')
    else:
        logger.info(f'phase: {phase.name} (skipped)')

    section_contents = LazyValue(
        functools.partial(
            load_section_contents,
            section_contents_file_path=section_contents_file_path,
        ))

    ## report
    phase = Phase.REPORT
    if phase in phases:
        logger.info(f'phase: {phase.name}')
        Settings.embed_model = embed_model.value
        generate_report(
            llm.value,
            index.value,
            outline.value,
            section_contents.value,
            result_file_path,
            introduction_instruction,
            conclusion_instruction,
        )
        logger.info('done')
    else:
        logger.info(f'phase: {phase.name} (skipped)')


    ## ragflow
    phase = Phase.RAGFLOW
    if phase in phases:
        logger.info(f'phase: {phase.name}')
        upload_to_ragflow(
            RAGFlow(api_key=ragflow_api_key, base_url=ragflow_base_url),
            ragflow_dataset_id,
            result_file_path,
        )
        logger.info('done')
    else:
        logger.info(f'phase: {phase.name} (skipped)')


if __name__ == '__main__':
    main()
