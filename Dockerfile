# Base Image
FROM python:3.10-slim-bullseye

# Install system dependencies
# gcc/g++ for compiling python extensions
# curl for installing rust
# libssl-dev/pkg-config for rust crates that might need ssl
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Update pip and install build tools
RUN pip install --no-cache-dir --upgrade pip maturin

# Build Rust Engine first (caching layer)
COPY rust_engine/ ./rust_engine/
WORKDIR /app/rust_engine
# Build and install the rust extension
# We use maturin to build a wheel and then install it
RUN maturin build --release --out dist && \
    pip install dist/*.whl

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# Attempt to handle potential encoding issues (UTF-16) by recreating the file if needed
# But standard pip might handle it. If it fails, we can add conversion.
# Using a simple python script to convert if needed before pip install
RUN python -c "try:\n    content = open('requirements.txt', 'rb').read().decode('utf-16')\n    with open('requirements_utf8.txt', 'w', encoding='utf-8') as f: f.write(content)\n    print('Converted UTF-16 to UTF-8')\nexcept:\n    print('File likely already UTF-8 or compatible')\n    import shutil; shutil.copy('requirements.txt', 'requirements_utf8.txt')" \
    && pip install --no-cache-dir -r requirements_utf8.txt

# Copy application code
COPY . .

# Expose the Flask/SocketIO port
EXPOSE 5000

# Environment Variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=web_dashboard.py

# Run the web dashboard
CMD ["python", "web_dashboard.py"]
