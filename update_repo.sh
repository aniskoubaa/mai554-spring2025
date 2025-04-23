#!/bin/bash

# update_repo.sh - Script to update the repository with version messages
# Usage: ./update_repo.sh "Your commit message"

# Exit immediately if a command exits with a non-zero status
set -e

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "Error: Please provide a commit message."
  echo "Usage: ./update_repo.sh \"Your commit message\""
  exit 1
fi

# Get the commit message from command line argument
COMMIT_MESSAGE="$1"

# Add version timestamp to commit message
VERSION_TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
FULL_MESSAGE="$COMMIT_MESSAGE (Version: $VERSION_TIMESTAMP)"

echo "Updating repository with message: $FULL_MESSAGE"

# Stage all changes
git add .

# Commit changes with the provided message
git commit -m "$FULL_MESSAGE"

# Push changes to remote repository
echo "Pushing changes to remote repository..."
git push

echo "Repository updated successfully!" 