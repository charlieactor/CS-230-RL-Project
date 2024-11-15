#!/bin/bash

export ZONE="us-east1-d"
export PROJECT_ID="" # blank for proprietary reasons
export INSTANCE_NAME="blan" # blank for proprietary reasons
export STARTUP_SCRIPT_PATH=./startup.sh
export SSH_PUBLIC_KEY_PATH=./data/gcloud_rsa.pub
export SSH_PRIVATE_KEY_PATH=./data/gcloud_rsa
export IMAGE_FAMILY="pytorch-latest-cpu"