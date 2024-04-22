import sys
sys.path.insert(1,'../repositories_used/ComboLoss/')

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.nets import ComboNet
import PIL

preprocess_beauty = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def load_beauty_model(pretrained_model_path, backbone='SEResNeXt50'):
    model = ComboNet(num_out=5, backbone_net_name=backbone)
    model = model.float()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device)

    if torch.cuda.is_available():
        print('------------------------------------')
        print('Loading beauty model on cuda')
        print('------------------------------------')
        model = model.cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(pretrained_model_path))
    else:
        print('------------------------------------')
        print('Loading beauty model on cpu')
        print('------------------------------------')
        model = model.to('cpu')
        state_dict = torch.load(pretrained_model_path,map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()
    return model

def calculate_beauty_score(img,model):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ## rgb img
        img = preprocess_beauty(PIL.Image.fromarray(img))
        img.unsqueeze_(0)
        img = img.to(device)

        score, cls = model(img)

        return round(float(score.to('cpu').detach().item()),2)