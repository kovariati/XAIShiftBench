FROM python:3.13.5-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /opt/xaishiftbench
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -r requirements-lock.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs
RUN python -m pip install --no-cache-dir --no-deps -e .
CMD ["python", "-m", "pytest", "-q", "-rs"]
