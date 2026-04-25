# 1. Use a lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy all your project files into the container
COPY . .

# 4. Install dependencies
RUN pip install --no-cache-dir .
RUN pip install -r requirements.txt

# 5. Make the startup script executable (Bypasses Windows permission errors)
RUN chmod +x apps/start.sh

# 6. Expose the port the main server uses
EXPOSE 8000

# 7. Start all services using the bash script
CMD ["./apps/start.sh"]