import numpy as np

class SemanticRouter:
    def __init__(self, embedding, routes):
        self.embedding = embedding
        self.routes = routes
        self.route_embeddings = {}

        for route in self.routes:
            self.route_embeddings[route.name] = self.embedding.encode(route.samples)

    def route(self, query):
        query_embedding = self.embedding.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores = []
        for route in self.routes:
            routes_embedding = self.route_embeddings[route.name] / np.linalg.norm(self.route_embeddings[route.name])
            score = np.mean(np.dot(routes_embedding, query_embedding.T).flatten())
            scores.append((score, route.name))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]