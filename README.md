

# Age-Gender Detection with FastAPI

This project is a FastAPI application that predicts age and gender from uploaded images. It utilizes OpenCV for face detection and neural networks for age and gender predictions.

## Prerequisites

- Docker

## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone Age-Gender-Detection
   cd Age-Gender-Detection
   ```

2. **Build the Docker image:**

   ```bash
   docker build -t fastapi_app .
   ```

3. **Run the container:**

   ```bash
   docker run -p 8000:8000 fastapi_app
   ```

4. After executing the above command, your FastAPI application should be running on `http://127.0.0.1:8000`.

## Usage

- Send a `POST` request to `http://127.0.0.1:8000/predict/` with an image file attached. The API will return the predicted age and gender.

  Example Response:
  ```json
  [
      {"gender": "Male", "age": "(25-32)"},
      {"gender": "Female", "age": "(15-20)"}
  ]
  ```

## Technology Stack

- **FastAPI**: Web framework to build APIs.
- **OpenCV**: Used for face detection.
- **Docker**: Used for containerization.

## Contributing

Feel free to submit pull requests or open issues to improve this project.

