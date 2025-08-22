import base64
import json
import logging
import os
import uuid
from typing import Annotated

import redis
import requests
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic.main import BaseModel
from starlette.background import BackgroundTasks

import segment_tiff

app = FastAPI()

log_level = os.getenv('LOG_LEVEL', 'DEBUG')
logging.basicConfig(level=log_level)

redisClient = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

current_processes = {}


class BlobPrediction(BaseModel):
    Algorithm: str = "fastSAM_GPU"
    BoxThreshold: float = 0.32
    TextThreshold: float = 0.32
    Blob: str = ""
    File: str = ""
    FileType: str = "jpeg"
    PhraseQuery: str = ""
    CenterCords: str = "0,0"
    downloadFileUrl: str = ""


@app.get("/")
async def predict_proxy():
    response = BlobPrediction()
    return response


"""
    Predict endpoint is used to start the analysis.
    @api {post} /_api/v1/predict
    @apiName Predict
    @apiGroup Analysis
    @apiVersion 1.0.0
    @apiDescription Predict endpoint is used to start the analysis.
    @apiParam {String} Algorithm The algorithm to be used for the analysis.
    @apiParam {Float} BoxThreshold The box threshold to be used for the analysis.
    @apiParam {Float} TextThreshold The text threshold to be used for the analysis.
    @apiParam {String} File The file to be analyzed.
    @apiParam {String} FileType The file type to be analyzed.
    @apiParam {String} PhraseQuery The phrase query to be used for the analysis.
    @apiParam {String} CenterCords The center cords to be used for the analysis.
    @apiSuccess {String} status The status of the analysis.
    @apiSuccess {String} analysis_id The analysis id.
    @apiSuccessExample Success-Response:
        HTTP/1.1 200 OK
        {
            "status": "Success",
            "chat_result_text": [],
            "output_image": "",
            "output_geojson": "",
            "output_tiff": ""
        }
"""


@app.post("/_api/v1/predict")
async def predict_proxy(blob_prediction: BlobPrediction):
    logging.debug("blob_prediction: %s", blob_prediction)

    if blob_prediction.File.startswith("http"):
        blob_prediction.downloadFileUrl = blob_prediction.File

    blob_prediction.File = "/local_upload_file." + blob_prediction.FileType
    blob_prediction.Algorithm = "fastSAM_GPU"
    analysis_id = uuid.uuid4().hex
    current_processes[analysis_id] = {
        "status": "Starting",
        "current_progress": "5.0",
        "current_state": "sending_analysis_request"
    }

    return send_analysis_request(blob_prediction, analysis_id)


"""
    This endpoint is used to upload a file to the server and start the analysis.
    @api {post} /_api/v1/predict_file
    @apiName PredictFile
    @apiGroup Analysis
    @apiVersion 1.0.0
    @apiDescription This endpoint is used to upload a file to the server and start the analysis.
    @apiParam {File} file The file to be uploaded.
    @apiParam {String} query_phrase The query phrase to be used for the analysis.
    @apiSuccess {String} status The status of the analysis.
    @apiSuccess {String} analysis_id The analysis id.
    @apiSuccessExample Success-Response:
        HTTP/1.1 200 OK
        {
            "status": "Started",
            "analysis_id": "1234567890"
        }
"""


@app.post("/_api/v1/predict_file")
def upload_file_to_server(file: Annotated[UploadFile, File()], query_phrase: Annotated[str, Form()],
                          FileType: Annotated[str, Form()],
                          background_tasks: BackgroundTasks):
    analysis_id = uuid.uuid4().hex

    logging.debug("form_data: %s", query_phrase)
    logging.debug("file: %s", file.filename)

    blob_prediction = BlobPrediction()
    blob_prediction.PhraseQuery = query_phrase
    blob_prediction.FileType = FileType
    blob_prediction.File = f"/demo/data/local_upload_file.{analysis_id}." + FileType
    open(blob_prediction.File, "wb").write(file.file.read())

    print(blob_prediction)

    current_processes[analysis_id] = {
        "status": "Starting",
        "current_progress": "5.0",
        "current_state": "sending_analysis_request"
    }

    background_tasks.add_task(send_analysis_request, blob_prediction, analysis_id)
    return {
        "status": "Started",
        "analysis_id": analysis_id
    }


@app.post("/_api/v1/predict_file_from_url")
def upload_file_to_server_from_url(file_url: Annotated[str, Form()], query_phrase: Annotated[str, Form()],
                                   file_type: Annotated[str, Form()],
                                   background_tasks: BackgroundTasks):
    analysis_id = uuid.uuid4().hex
    logging.debug("query_phrase: %s", query_phrase)
    logging.debug("file_url: %s", file_url)
    blob_prediction = BlobPrediction()
    blob_prediction.PhraseQuery = query_phrase
    blob_prediction.FileType = file_type

    current_processes[analysis_id] = {
        "status": "Starting",
        "current_progress": "5.0",
        "current_state": "sending_analysis_request"
    }

    blob_prediction.File = "./local_upload_file." + blob_prediction.FileType
    blob_prediction.Algorithm = "fastSAM_GPU"

    background_tasks.add_task(send_analysis_request, blob_prediction, analysis_id)
    return {
        "status": "Started",
        "analysis_id": analysis_id
    }


def send_analysis_request(blob_prediction: BlobPrediction, analysis_id: str = None):
    if blob_prediction.downloadFileUrl != "":
        logging.debug("Downloading file from url: %s", blob_prediction.downloadFileUrl)
        r = requests.get(blob_prediction.downloadFileUrl, allow_redirects=True)
        open(blob_prediction.File, 'wb').write(r.content)
        blob_prediction.Blob = blob_prediction.File

    try:
        current_processes[analysis_id] = {
            "status": "Starting Analysis",
            "current_progress": "35.0",
            "current_state": "starting_analysis"
        }

        current_processes[analysis_id] = {
            "status": "Analysis In Progress",
            "current_progress": "35.0",
            "current_state": "analysis_in_progress"
        }

        result = segment_tiff.predict(blob_prediction.Algorithm, blob_prediction.BoxThreshold,
                                      blob_prediction.TextThreshold, blob_prediction.Blob, blob_prediction.PhraseQuery,
                                      blob_prediction.CenterCords, "")

        logging.info(f">> [{analysis_id}] object finished analysis, returning back data. << ")
        print(result)
        output_geojson = None
        output_tiff = None
        output_image = None

        chat_result_text = result[0]
        if result[1] is not None:
            output_image = base64.b64encode(open(result[1], "rb").read()).decode('utf-8')

        if result[2] is not None:
            output_geojson = open(result[2], "rb").read().decode(errors='replace')

        if result[3] is not None:
            output_tiff = base64.b64encode(open(result[3], "rb").read()).decode('utf-8')

        response_object = {
            "status": "Success",
            "chat_result_text": chat_result_text,
            "output_image": output_image,
            "output_geojson": output_geojson,
            "output_tiff": output_tiff
        }
    except Exception as e:
        logging.error(f"Error while processing the object. {e.__str__()}")
        print(e)
        response_object = {
            "status": "Error",
            "chat_result_text": [],
            "output_image": "",
            "output_geojson": "",
            "output_tiff": ""
        }

    if analysis_id is not None:
        redisClient.set(analysis_id, json.dumps(response_object), ex=3600)
        logging.info(f"Object results saved in Redis for further use. {analysis_id}")
    return response_object


"""
    This endpoint is used to check the status of the analysis, should be check by the client periodically.
    @api {get} /_api/v1/analysis_status/:analysis_id
    @apiName GetAnalysisStatus
    @apiGroup Analysis
    @apiVersion 1.0.0
    @apiDescription This endpoint is used to check the status of the analysis. Checked by the client periodically.
    @apiParam {String} analysis_id The analysis id.
    @apiSuccess {String} status The status of the analysis.
    @apiSuccessExample Success-Response:
        HTTP/1.1 200 OK
        {
            "status": "Success",
            "chat_result_text": [],
            "output_image": "",
            "output_geojson": "",
            "output_tiff": ""
        }
    @apiErrorExample Error-Response:
        HTTP/1.1 404 Not Found
        {
            "status": "NotFound"
        }
        
"""


@app.get("/_api/v1/analysis_status/{analysis_id}")
async def background_analysis(analysis_id: str):
    if analysis_id not in current_processes:
        return {
            "status": "Preparing"
        }

    response = current_processes[analysis_id]
    analysis_entity = redisClient.get(analysis_id)
    if analysis_entity is not None:
        response = json.loads(analysis_entity)
        logging.debug("Analysis object received, %s", response["chat_result_text"])

    return response


origins = ["*"]

if __name__ == "__main__":
    segment_tiff.load_model()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=log_level.lower())
