# One container that runs the whole system: the API, which also serves the
# built frontend. Used for hosting (see README, "Hosting it online").
#
#   docker build -t sci-rtw .
#   docker run -p 8000:8000 -e SCI_RTW_ADMIN_PASSWORD=choose-one sci-rtw
#   then open http://localhost:8000

# ---- Stage 1: build the React frontend ----
FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: the API ----
FROM python:3.14-slim
WORKDIR /app

# Only what the API needs at run time (requirements.txt also lists the
# training and plotting libraries).
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY models/ ./models/
COPY --from=frontend /frontend/dist ./frontend/dist

ENV SCI_RTW_FRONTEND_DIR=/app/frontend/dist \
    SCI_RTW_DB=/app/data/sci_rtw.db
RUN mkdir -p /app/data

WORKDIR /app/api
EXPOSE 8000
# Hosting services pass the port to listen on as $PORT.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
