FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Expose ports
EXPOSE 9002 9502
# Run both apps (FastAPI & Streamlit) together
COPY start.sh .
CMD ["./start.sh"]

