## File to generate and save feature maps for view in the interface

from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision
import torch

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

thisPath = ""

def generate_feature_maps(img, con_layer):
    """
    Generate feature maps for a given image and convolutional layer number (1-17)

    Args:
        img (PIL): The image passed in by the user
        con_layer (int): The number corresponding to the convolutional layer to show

    Returns:
        str: The path to the generated feature map image
    """
    this_img = np.array(img)
    image = Image.fromarray(this_img, 'RGB')
    plt.imshow(image)

    # model = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    # Plot and save feature maps for each layer
    for i, (fm, name) in enumerate(zip(processed, names)):
        fig = plt.figure(figsize=(10, 10))
        a = fig.add_subplot(1, 1, 1)  # You should adjust the layout as needed
        imgplot = plt.imshow(fm, cmap='viridis')  # Adjust the colormap if needed
        a.axis("off")
        filename = f'layer{i}.jpg'
        plt.savefig("outputs\\runs\\detect\\exp\\layers\\" + filename, bbox_inches='tight')
        plt.close(fig)  # Close the figure after saving
    
    this_dir = "outputs\\runs\\detect\\exp\\layers\\layer" + str(int(int(con_layer) - 1)) + '.jpg'
    print("Convolutional layers Generated")
    return this_dir