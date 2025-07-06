# Automated Detection and Classification of Bone Fractures Using Deep Learning (ResNet18)

A system to detect and classify bone fractures in X-ray images using ResNet18, featuring saliency maps for interpretability and a web interface with an API for predictions.

## Dataset
- *Name*: Custom X-ray Fracture Dataset
- *Source*: https://drive.google.com/drive/folders/1lDTGdXK5kYpPfP_-Y8LhGyIubs8Lcprg?usp=drive_link
- *Description*: Contains 4159 X-ray images with fracture classification
## Platform
- Python 3.8+ on Jupyter Notebook or VS Code
- Front-end: HTML5, CSS3, JavaScript
- Dependencies: Listed in requirements.txt (torch, torchvision, scikit-learn, matplotlib, fastapi, uvicorn)

## How to Run
1. Clone the repository: https://github.com/akirijanisrael001/Automated_fractureDetection
2. Set up a virtual environment:
   - Windows: python -m venv venv then venv\Scripts\activate
   - Mac/Linux: python3 -m venv venv then source venv/bin/activate
3. Install dependencies: pip install -r requirements.txt
4. Download the dataset from the Google Drive link and place it in /data/train and /data/test folders.
5. Train the model: python code/main.py (runs for 18 epochs, saves model as fracture_model.pth)
6. Start the API server: python api.py
7. Run the front-end: Open frontend/index.html in a browser or use python -m http.server in /frontend

## Expected Output
- *Model*: Prints accuracy (e.g., ~81.4%), loss per epoch, generates saliency maps, saves ROC curve (roc_curve.png), and model weights (fracture_model.pth).
- *API*: Returns JSON with prediction (e.g., "fracture" or "no_fracture") and confidence score when accessed via /predict endpoint.
- *Front-end*: Displays navbar, appointment booking form (date/time inputs), upload page with prediction results from the API.
