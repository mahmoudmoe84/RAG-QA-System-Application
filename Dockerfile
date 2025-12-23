#============================
# RAG Q&A Application Dockerfile
#============================


#build stage 
FROM python:3.13-slim AS builder

# set working directory
WORKDIR /app

# install system dependencies
#this command installs the build-essential package, which includes a collection of tools for compiling and building software.
RUN apt-get update && apt-get install -y  --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements file
COPY requirements.txt .

# create a virtual environment and install python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# production stage 
FROM python:3.13-slim AS production

# set working directory
WORKDIR /app

#install git (required by RAGAS/databsets library)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# create non root user for security best practices
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

#copy application code
COPY . .
# first . in the copy command refers to the current directory on the host machine (where the Dockerfile is located),
# and the second . refers to the /app directory inside the Docker container.

#set ownership to non-root user 
RUN chown -R appuser:appgroup /app

#switch to non-root user
USER appuser

# set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app 

# expose port
EXPOSE 8000

#health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
 CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]