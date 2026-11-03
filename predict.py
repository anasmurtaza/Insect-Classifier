from pathlib import Path
from PIL import Image,ImageOps
import torch
from data import transform
from model import load_model
from config import MODEL_PATH,CLASSES
class Predictor:
    def __init__(self,path=MODEL_PATH):
        if not Path(path).is_file():raise FileNotFoundError('No trained model. First run: python train.py --epochs 30')
        self.model=load_model(path);self.transform=transform(False)
    def predict(self,path):
        with Image.open(path) as image:
            tensor=self.transform(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)
        with torch.inference_mode():scores=torch.softmax(self.model(tensor),1)[0]
        return sorted(zip(CLASSES,scores.tolist()),key=lambda p:p[1],reverse=True)
