import torch
from torch import nn
import copy
import torchvision
from torchvision import transforms as T
import gradio as gr


def launch(model_path):
    '''
    Launch the user interface that provides the predictions and corresponding
    class probabilities based on trained models from input model file path.

    Input:
        model_path: String
    
    Output:
        None (will launch the user interface)
    '''
    
    # load model from model file path
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    best_model_wts = copy.deepcopy(checkpoint)
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(best_model_wts)

    # prediction function
    def model_pred(img):
        class_names = ['artificial','human']
        transform=T.Compose([   
            T.Resize(256),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor=transform(img).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred_probs = torch.softmax(model(image_tensor), dim=1)
            pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
        return pred_labels_and_probs
    
    inputs = gr.Image(type="pil")
    interface = gr.Interface(
        fn=model_pred, 
        inputs=inputs, 
        outputs=gr.Label(num_top_classes=2, label="Predictions"), 
        title="Image Classification Prediction",
        description="Provide an image and get the predicted class label.")
    interface.launch()

if __name__ == "__main__":
    model_path = "models/model1.pth"
    launch(model_path)
