import torch
import torchvision.transforms as T
import numpy as np


class DINOV2Adapter():
    """Adapter for DINOv2 models"""
    def __init__(self):
        super().__init__()
        # Hardcoded settings for small DINOv2 model
        backbone_name = "dinov2_vits14"  # small DINOv2 with ViT-S/14
        self.DINO_input_size = 1400 # Standard input size for small model
        
        self.model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)

    def extract(self, image):
        """Extract features from an image"""
        
        self.model.eval()
        self.model.cuda()

        img_tensor = self._transform_input(image)
        # Extract features from the model
        with torch.no_grad():
            features = self.model.get_intermediate_layers(img_tensor, n=1)
            feature_tensor = features[-1]
            

        return self._transform_output(feature_tensor)
    
    def _transform_input(self, image):
        """Transform input image to the format that the model expects"""
        transform = T.Compose([
            T.Resize(self.DINO_input_size),  # Resize to 1400 (DINO's expected input size)
            T.CenterCrop(self.DINO_input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0).cuda()
        return img_tensor
    
    def _transform_output(self, raw_features):
        """Transform raw features to the format that the rest of the system expects"""
        features_np = raw_features.squeeze(0).cpu().numpy() 

        num_tokens = features_np.shape[0]
        if int(np.sqrt(num_tokens))**2 != num_tokens:
            features_np = features_np[1:]
            num_tokens = features_np.shape[0]
        return features_np