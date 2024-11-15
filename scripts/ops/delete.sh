#!/bin/bash

source ./config.sh

gcloud compute instances delete $INSTANCE_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --quiet
