#!/bin/bash

source ./config.sh

# Create the instance with the startup script and SSH key
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --image-family=$IMAGE_FAMILY \
    --image-project=deeplearning-platform-release \
    --machine-type=c3-standard-176

