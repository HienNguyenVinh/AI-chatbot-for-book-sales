from Embeddings import SentenceTransformerEmbedding
import pymongo
from IPython.display import Markdown
import textwrap

class RAG:
    def __init__(self,
                  mongodbUrl: str,
                  dbName: str,
                  dbCollection: str,
                  llm: str,
                  embeddingName: str='keepitreal/vietnamese-sbert'):
        self.mongo_client = self.get_mongo_client(mongodbUrl)
        self.db = self.mongo_client[dbName]
        self.collection = self.db[dbCollection]
        self.embedding_model = SentenceTransformerEmbedding(embeddingName)
        self.llm = llm

    def get_mongo_client(self, mongo_url):
        try:
            client = pymongo.MongoClient(mongo_url, appname='devrel.content.python')
            print("Connected to MongoDB successfully!")
            return client
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print(f"Error connecting to MongoDB: {e}")
            return None

    def get_embedding(self, text):
        if not text.strip():
            # print('Attemped to get embedding for empty text.')
            return []

        return self.embedding_model.encode(text).tolist()

    def vector_search(self, user_query: str, limit=4):
        query_embedding = self.get_embedding(user_query)

        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        # vector search pipeline
        vector_search_stage = {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 100,
                    "limit": limit
                }
        }
        unset_stage = {
            '$unset': 'embedding'    # exclude the 'embedding' field from the results
        }
        project_state = {
            '$project': {
                '_id': 0,    # exclued the _id field
                'name': 1,
                'author': 1,
                'category': 1,
                'price': 1,
                'full_description': 1,
                'score': {'$meta': 'vectorSearchScore'},
            }
        }

        pipeline = [vector_search_stage, unset_stage, project_state]
        results = self.collection.aggregate(pipeline)

        return list(results)

    def get_full_prompt(self, query):
        results = self.vector_search(query, 5)

        full_prompt = ""
        i = 0
        for result in results:
            if result.get('name'):
                i += 1
                full_prompt += f"\n {i} Tên: {result.get('name')}"
                if result.get('author'):
                    full_prompt += f", Tác giả: {result.get('author')}"
                if result.get('category'):
                    full_prompt += f", Thể loại: {result.get('category')}"
                if result.get('price'):
                    full_prompt += f", Giá: {result.get('price')}"
                else:
                    full_prompt += f", Giá: Liên hệ để trao đổi"

                full_prompt += ", Mô tả: {result.get('full_description')}"

        return full_prompt

    def get_response(self, prompt):
        return self.llm.generate_content(prompt)

    def _to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))