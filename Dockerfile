FROM python:3.11-slim AS builder
RUN pip install uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt