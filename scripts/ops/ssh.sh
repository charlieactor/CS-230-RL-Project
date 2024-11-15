#!/bin/bash

source ./config.sh

if [[ $1 == "--setup" ]]; then
    # gcloud compute config-ssh --ssh-key-file=$SSH_PRIVATE_KEY_PATH
    # git remote set-url gcloud $INSTANCE_NAME.$ZONE.$PROJECT_ID:~/director
    gcloud compute scp ./setup.sh $INSTANCE_NAME:~/ --zone=$ZONE --project=$PROJECT_ID
    gcloud compute scp ../../.env $INSTANCE_NAME:~/ --zone=$ZONE --project=$PROJECT_ID
    gcloud compute scp ~/.ssh/id_ed25519* $INSTANCE_NAME:~/.ssh/ --zone=$ZONE --project=$PROJECT_ID
fi

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
