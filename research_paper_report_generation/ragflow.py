from uuid import uuid4


def upload_to_ragflow(rag_object, dataset_id, document_path):
    dataset = rag_object.list_datasets(id=dataset_id)[0]

    filename = f'{uuid4()}.md'

    with open(document_path, 'rb') as f:
        content = f.read()

    dataset.upload_documents([{
        'display_name': filename,
        'blob': content,
    }])

    document = dataset.list_documents(keywords=filename, page_size=1)[0]

    dataset.async_parse_documents([document.id])
