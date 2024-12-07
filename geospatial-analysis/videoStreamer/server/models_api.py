import os

from videoStreamer.server.base import Base

"""
@api {get} /models/get_all Get All Trained Models
@apiName GetTrainedModels
@apiGroup Models
@apiVersion 1.0.0
@apiDescription This endpoint returns all trained models
@apiSuccess {String} status The status of the request
@apiSuccess {String[]} models The list of trained models
@apiSuccessExample {json} Success-Response:
    HTTP/1.1 200 OK
    {
        "status": "success",
        "models": ["model1", "model2"]
    }
"""


class TrainedModelsRequest(Base):
    def __init__(self):
        super().__init__()
        self.local_path_models = []

    def get_local_path_models(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parent_dir = os.path.join(parent_dir, '../')
        print(parent_dir)
        models = [f for f in os.listdir(parent_dir) if f.endswith('.pt')]
        self.local_path_models = parent_dir
        return models

    def invoke(self):
        local_models = self.get_local_path_models()
        return {
            "status": "success",
            "models": local_models
        }
