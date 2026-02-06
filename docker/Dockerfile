name: Docker Build & Push

on:
  push:
    branches: [main, develop]
    tags:
      - "v*.*.*"
  pull_request:
    branches: [main, develop]
  workflow_dispatch:

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      # 1️⃣ Checkout repo
      - name: Checkout repository
        uses: actions/checkout@v4

      # 2️⃣ Set up Docker Buildx
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      # 3️⃣ Log in to Docker Hub (skip for PRs)
      - name: Log in to Docker Hub
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      # 4️⃣ Build and push Docker image
      - name: Build and push Docker image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile            # ✅ Correct Dockerfile path
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ secrets.DOCKER_USERNAME }}/covenant-ai:main
          cache-from: type=gha
          cache-to: type=gha,mode=max

      # 5️⃣ Optionally, load image locally (for future use in jobs)
      - name: Load Docker image locally
        if: github.event_name == 'pull_request'
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile
          load: true
