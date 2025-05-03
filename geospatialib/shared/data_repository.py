from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


class MilvusRepository:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.client = connections.connect("default", host="localhost", port="19530")
        self.schema = None
        self.collection = self._get_collection()

    def _get_collection(self):
        if utility.has_collection(self.client, self.collection_name):
            return Collection(self.collection_name, self.client)
        else:
            self._create_collection()
            return self.collection

    def _create_collection(self):
        schema = CollectionSchema(fields=self.schema, description="collection for face recognition")
        collection_instance = Collection(name=self.collection_name, schema=schema, client=self.client)
        self.collection = collection_instance
        print(f"Created collection {self.collection_name}")

    def create_index(self, name, index_type, metric_type, params):
        index = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": params,
        }
        self.collection.create_index(name, index)
        print(f"Created index {name} for collection {self.collection_name}")

    def insert(self, data):
        self.collection.insert(data)

    def search(self, data, top_k):
        return self.collection.search(data, top_k=top_k)

    def __del__(self):
        self.client.close()
