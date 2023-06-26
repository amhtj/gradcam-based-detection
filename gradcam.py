import cv2
import numpy as np

import sklearn.metrics
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_gradcam_layer(model):
        #model-specific function to get the most suitable layer 
        #for the gradcam (usually the last conv one)
        layer = model.layer4[-1].conv3
        return layer
    
activation_maps = None

def get_activation_maps(module, input, output):
    # Store the gradients of the specific feature map
    global activation_maps
    activation_maps = output
    activation_maps.retain_grad()

class GradcamBBoxPredictor():
    def __init__(self, gradcam_relative_threshold=0.5, class_threshold=0.01, use_tta=True):
        
        self.gradcam_relative_threshold = gradcam_relative_threshold
        self.class_threshold = class_threshold
        self.use_tta = use_tta
        n_classes = 20
        self.model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self. model.fc = nn.Linear(num_ftrs, n_classes)

        checkpoint_state = torch.load('model_at_epoch_019_compatible.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint_state['model_sdict'])
        self.model.eval();
        self.model.to(device);
        self.gradcam_layer = get_gradcam_layer(self.model)
        forward_hook = self.gradcam_layer.register_forward_hook(get_activation_maps)
        
    def get_gradcam_map_and_class_probabilities(self, input_data, class_number):
        out = self.model(input_data)
        class_probabilities = torch.nn.functional.sigmoid(out)[:,class_number]
        loss = out[:, class_number].sum()
        loss.backward()

        grad_coefs = activation_maps.grad.mean((2,3), keepdims=True)
        gradcam_maps = torch.nn.functional.relu(grad_coefs*activation_maps).sum((1))

        return gradcam_maps, class_probabilities
    
    def extract_bboxes_from_gradcam_map(self, gradcam_map, relative_threshold=0.15):
        threshold = gradcam_map.max()*relative_threshold
        gradcam_binarized = cv2.threshold(gradcam_map, thresh=threshold,
                                          maxval=1, type=cv2.THRESH_BINARY)[1].astype(np.uint8)
        num_comps, connectivity, stats, centroids = cv2.connectedComponentsWithStats(gradcam_binarized)
        stats = stats[1:,] #omitting background
        #converting to voc format
        x, y, w, h = stats[:,cv2.CC_STAT_LEFT], stats[:,cv2.CC_STAT_TOP], stats[:,cv2.CC_STAT_WIDTH], stats[:,cv2.CC_STAT_HEIGHT]
        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h
        voc_bboxes = np.vstack([xmin, ymin, xmax, ymax]).transpose()
        return voc_bboxes
    
    def extract_bboxes_from_sample(self, sample, class_number=4):
        input_tensor = sample[0].unsqueeze(0).to(device)
        input_image = sample[2]['image']
        gradcam_maps, class_probabilities = self.get_gradcam_map_and_class_probabilities(input_tensor, class_number)
        if self.use_tta:
            gradcam_maps_flipped, class_probabilities_flipped = self.get_gradcam_map_and_class_probabilities(torch.fliplr(input_tensor), class_number)
            gradcam_maps += torch.fliplr(gradcam_maps_flipped)
            class_probabilities += class_probabilities_flipped
            gradcam_maps /= 2.
            class_probabilities /= 2.
        
        class_probability = class_probabilities[0].item()
        if class_probability > self.class_threshold:
            sample_gradcam_map = gradcam_maps[0].detach().cpu().numpy()
            gradcam_heatmap = cv2.resize(sample_gradcam_map,
                                         (input_image.shape[1], input_image.shape[0]),
                                         interpolation=cv2.INTER_CUBIC)
            bboxes = self.extract_bboxes_from_gradcam_map(gradcam_heatmap, self.gradcam_relative_threshold)
            #adding confidence column
            ans = np.zeros((bboxes.shape[0], 5))
            ans[:,:4] = bboxes
            ans[:,4] = class_probability
            return ans
        else:
            return np.array([])
  


        
def produce_gradcam_bboxes(dataset, bbox_predictor, class_number=4):
    gradcam_bboxes = {}
    for sample in dataset:
        key = sample[2]['impath'].split('/')[-1]
        bboxes = bbox_predictor.extract_bboxes_from_sample(sample, class_number)
        gradcam_bboxes[key] = bboxes
    return gradcam_bboxes
    
    

    


