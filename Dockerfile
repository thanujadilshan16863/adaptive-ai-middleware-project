FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-chronos.txt ./
ARG WITH_CHRONOS=false
RUN if [ "$WITH_CHRONOS" = "true" ]; then pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-chronos.txt; else pip install --no-cache-dir -r requirements.txt; fi
COPY backend ./backend
RUN useradd --create-home worker && mkdir /app/state && chown worker:worker /app/state
USER worker
ENV DATABASE_URL=sqlite:////app/state/middleware.db
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
