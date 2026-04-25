#!/usr/bin/env bash
# setup_hdfs.sh
# -------------
# Create required HDFS directories after the cluster starts.
# Waits for the namenode to be ready before issuing mkdir commands.
#
# Usage:
#   bash scripts/setup_hdfs.sh [namenode_host] [namenode_port]
#
# Defaults:
#   namenode_host = hdfs-namenode (Docker service name)
#   namenode_port = 9870          (HDFS Web UI / REST port)

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────

NAMENODE_HOST="${1:-hdfs-namenode}"
NAMENODE_PORT="${2:-9870}"
HDFS_URI="hdfs://${NAMENODE_HOST}:9000"

NAMENODE_URL="http://${NAMENODE_HOST}:${NAMENODE_PORT}"
MAX_WAIT_S=120   # Maximum seconds to wait for the namenode
POLL_INTERVAL=5  # Seconds between readiness checks

# HDFS directories to create
HDFS_DIRS=(
    "/data/flights"
    "/data/flights/cleaned"
    "/data/flights/featured"
    "/models"
    "/models/preprocessing_pipeline"
    "/output/streaming_predictions"
    "/output/batch_predictions"
    "/output/eda"
    "/checkpoints/streaming"
    "/tmp"
)

# ─── Logging helpers ──────────────────────────────────────────────────────────

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  $*"; }
warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]  $*" >&2; }
err()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; }

# ─── Wait for namenode ────────────────────────────────────────────────────────

wait_for_namenode() {
    log "Waiting for HDFS namenode at ${NAMENODE_URL} …"
    local elapsed=0

    until curl -sf "${NAMENODE_URL}/jmx?qry=Hadoop:service=NameNode,name=NameNodeStatus" \
           | grep -q '"State":"active"' 2>/dev/null; do
        if (( elapsed >= MAX_WAIT_S )); then
            err "Namenode not ready after ${MAX_WAIT_S}s. Aborting."
            exit 1
        fi
        log "Namenode not ready yet. Retrying in ${POLL_INTERVAL}s … (${elapsed}/${MAX_WAIT_S}s)"
        sleep "${POLL_INTERVAL}"
        (( elapsed += POLL_INTERVAL ))
    done

    log "Namenode is active and ready."
}

# ─── Helper: run hdfs dfs command ─────────────────────────────────────────────

hdfs_cmd() {
    # Try docker exec into the namenode container first; fall back to local hdfs command.
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^hdfs-namenode$"; then
        docker exec hdfs-namenode hdfs "$@"
    elif command -v hdfs &>/dev/null; then
        hdfs "$@"
    else
        err "Cannot find 'hdfs' command and no namenode container is running."
        exit 1
    fi
}

# ─── Create HDFS directories ──────────────────────────────────────────────────

create_hdfs_directories() {
    log "Creating HDFS directory structure …"

    for dir in "${HDFS_DIRS[@]}"; do
        if hdfs_cmd dfs -test -d "${dir}" 2>/dev/null; then
            log "Directory already exists: ${dir}"
        else
            log "Creating: ${dir}"
            hdfs_cmd dfs -mkdir -p "${dir}"
        fi
    done

    log "Setting permissions …"
    hdfs_cmd dfs -chmod -R 777 /data
    hdfs_cmd dfs -chmod -R 777 /models
    hdfs_cmd dfs -chmod -R 777 /output
    hdfs_cmd dfs -chmod -R 777 /checkpoints
    hdfs_cmd dfs -chmod -R 777 /tmp

    log "Directory listing:"
    hdfs_cmd dfs -ls /
}

# ─── Verify directories ───────────────────────────────────────────────────────

verify_directories() {
    log "Verifying HDFS directories …"
    local failed=0

    for dir in "${HDFS_DIRS[@]}"; do
        if hdfs_cmd dfs -test -d "${dir}" 2>/dev/null; then
            log "  [OK] ${dir}"
        else
            warn "  [MISSING] ${dir}"
            (( failed++ ))
        fi
    done

    if (( failed > 0 )); then
        err "${failed} directory/directories could not be verified."
        exit 1
    fi

    log "All HDFS directories verified successfully."
}

# ─── Main ─────────────────────────────────────────────────────────────────────

main() {
    log "HDFS setup script starting."
    log "  Namenode : ${NAMENODE_HOST}:${NAMENODE_PORT}"
    log "  HDFS URI : ${HDFS_URI}"

    wait_for_namenode
    create_hdfs_directories
    verify_directories

    log "HDFS setup completed successfully."
}

main "$@"
