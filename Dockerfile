# Stage 1: Build the React frontend
FROM node:18 AS frontend

# Set the working directory for the frontend
WORKDIR /frontend

# Copy package.json and package-lock.json to install dependencies
COPY package.json package-lock.json ./
RUN npm install --legacy-peer-deps

# Copy the rest of the frontend files and build the React app
COPY ./ ./
RUN npm run build || { echo "React build failed"; exit 1; }

# Stage 2: Set up the Python backend
FROM python:3.9-slim AS backend

# Set the working directory for the backend
WORKDIR /app

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.org/simple -r requirements.txt

# Copy the backend code
COPY backend/ .

# Copy the built React frontend into the backend
COPY --from=frontend /frontend/dist /app/frontend

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the FastAPI app
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

########################## AMD SERVER VERSION ########################### Stage 1: Build the React frontend
