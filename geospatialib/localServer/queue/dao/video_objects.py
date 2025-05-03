from pymilvus import FieldSchema, DataType

from shared.data_repository import MilvusRepository


class VideoObject(MilvusRepository):
    def __init__(self):
        self.schema = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="random", dtype=DataType.DOUBLE),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
        ]
        super().__init__("video_object")
        self._create_index()

    def _create_index(self):
        super().create_index(
            name="video_object_index",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 128})
