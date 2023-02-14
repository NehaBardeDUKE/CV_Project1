import torch
from torch import nn
import copy
import torchvision
from torchvision import transforms
from PIL import Image
import gradio as gr
from gradio.components import Image, Textbox,Label
import os
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    best_model_wts = copy.deepcopy(checkpoint)
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(best_model_wts)
    return model
def model_pred(img):
    transform=transforms.Compose([   
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor=transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred_probs = torch.softmax(model(image_tensor), dim=1)
        pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    return pred_labels_and_probs
model_path = 'model1.pth'
model = load_model(model_path)
class_names = ['artificial','human']
inputs = gr.Image(type="pil")
example_list = [["examples/" + example] for example in os.listdir("examples")]

interface = gr.Interface(
    fn=model_pred, 
    inputs=inputs, 
    outputs=gr.Label(num_top_classes=2, label="Predictions"), 
    title="Image Classification Prediction",
    description="Provide an image and get the predicted class label.",
    examples=example_list)

interface.launch()