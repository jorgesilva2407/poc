#!/bin/bash
set -e

if [ -z "$BUCKET" ]; then
  printf "Error: BUCKET environment variable is not set."
  exit 1
fi

printf "Executing dry run for syncing data to gs://$BUCKET/data\n\n"

# Dry run to check what would be synced
gcloud storage rsync -r --delete-unmatched-destination-objects --dry-run data/processed gs://$BUCKET/data

# Prompt for confirmation
printf "\n\nAre you sure you want to sync the data to gs://$BUCKET/data? (y/n): " 
read confirm
if [[ "$confirm" != "y" ]]; then
  printf "Sync operation cancelled."
  exit 0
fi

printf "Syncing data to gs://$BUCKET/data\n\n"

# Sync raw data to GCS
gcloud storage rsync -r --delete-unmatched-destination-objects data/processed gs://$BUCKET/data
