# Dataset Cluster Tool

## Overview

This application clusters images and provides a web interface for selecting representative images from each cluster.

## Project Structure

- `backend/`: Contains server code and clustering script.
  - `scripts/`: Python script for clustering images.
  - `data/`: Contains `dataset/`, `clusters/`, and `exported_images/`.
- `frontend/`: React application for the web interface.

## Setup Instructions

### Prerequisites

- Node.js and npm
- Python 3.x and pip

### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Install Node.js dependencies:

   ```bash
   npm install
   ```

3. Install Python dependencies:

   ```bash
   pip install -r scripts/requirements.txt
   ```

4. Prepare your dataset:

   - Place your images in `backend/data/dataset/`.

5. Run the clustering script:

   ```bash
   npm run cluster
   ```

6. Start the server:

   ```bash
   npm start
   ```

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the React application:

   ```bash
   npm start
   ```

4. Open your browser and navigate to `http://localhost:3000`.

## Usage

- Use the web interface to view image clusters.
- Click on images to select or deselect them.
- Click the "Export" button to save selected images to `backend/data/exported_images/`.

## License

[MIT License](LICENSE)