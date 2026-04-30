#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

REGISTRY="${REGISTRY:-ghcr.io/systemicvoid}"
IMAGE="${IMAGE:-${REGISTRY}/h-neurons-runpod}"

INPUT_HASH="$(
    sha256sum \
        requirements.txt \
        pyproject.toml \
        scripts/infra/docker/Dockerfile \
        scripts/infra/docker/Dockerfile.dockerignore \
        scripts/infra/docker/start.sh \
        scripts/infra/docker/smoke_triton.py \
    | sha256sum \
    | cut -c1-12
)"
GIT_SHA="$(git rev-parse --short HEAD)"
TAG="${INPUT_HASH}-${GIT_SHA}"

docker buildx build \
    --platform=linux/amd64 \
    --file scripts/infra/docker/Dockerfile \
    --tag "${IMAGE}:${TAG}" \
    --tag "${IMAGE}:latest" \
    --push \
    .

DIGEST="$(
    docker buildx imagetools inspect "${IMAGE}:${TAG}" \
        --format '{{.Manifest.Digest}}'
)"
printf '%s\n' "${DIGEST}" | tee /tmp/h-neurons-runpod-digest
printf '\n'
printf 'Image: %s:%s\n' "${IMAGE}" "${TAG}"
printf 'Profile pin: %s@%s\n' "${IMAGE}" "${DIGEST}"
printf 'Update runpod.expected_image_digest in scripts/infra/cloud/profiles/*-runpod.toml to: %s\n' "${DIGEST}"
