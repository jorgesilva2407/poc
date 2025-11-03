#!/bin/bash
set -e

if [ -z "${IMAGE_URI:-}" ]; then
    printf "IMAGE_URI is not set. Please set the IMAGE_URI environment variable."
    exit 1
fi

gcloud builds submit --tag "$IMAGE_URI" .
