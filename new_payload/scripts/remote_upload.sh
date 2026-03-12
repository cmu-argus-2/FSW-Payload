#!/bin/bash
set -e  # Exit on error

# === Configuration ===
SRC_DIR="/home/hazcoper/schl/sat/argus/new_payload/"
DEST_USER="argus-payload"
DEST_HOST="172.20.70.19"
DEST_PATH="/home/argus-payload/new_payload/"
IGNORE_FILE=".rsyncignore"

# === Sync via rsync ===
echo "Syncing $SRC_DIR to $DEST_USER@$DEST_HOST:$DEST_PATH ..."
rsync -avz --itemize-changes \
  --exclude-from="$IGNORE_FILE" \
  --checksum \
  --no-owner --no-times --no-perms --no-group \
  "$SRC_DIR" \
  "$DEST_USER@$DEST_HOST:$DEST_PATH" \
  --info=stats2


# === Run remote command ===
# REMOTE_CMD="cd $DEST_PATH && ./run.sh"
# echo "Running remote command: $REMOTE_CMD"
# ssh "$DEST_USER@$DEST_HOST" "$REMOTE_CMD"

# echo "Done!"
