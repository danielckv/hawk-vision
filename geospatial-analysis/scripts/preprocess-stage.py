#sh /usr/src/ultralytics/ultralytics/yolo/data/scripts/download_weights.sh yolov8n.pt --dockerfile already copies this 
python ultralytics/yolo/v8/detect/train.py --data ./dataset --weights /usr/src/ultralytics/ultralytics/yolov8n
yolo 

yolo \
  --task=$TASK \
  --mode=$MODE \
  --args=$ARGS \
  --training-images-location=$TRAININGIMAGES_LOCATION \
  --training-bucket=$TRAINING_BUCKET \
  --batch-size=$BATCH_SIZE \

kubectl -n presight-ai-test-01 exec -it $(kubectl -n presight-ai-test-01 get pods -l app=video-yolov8 -o name) -- /bin/bash -c "yolo \
--task='$TASK' \
--mode='$MODE' \
--args='$ARGS' \
--training-images-location='$TRAININGIMAGES_LOCATION' \
--training-bucket='$TRAINING_BUCKET' \
--batch-size='$BATCH_SIZE'"

kubectl -n presight-ai-test-01 exec -it $(kubectl -n presight-ai-test-01 get pods -l app=video-yolov8 -o name) -- /bin/bash
