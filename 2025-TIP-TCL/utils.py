import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

def save_config(opt, file_path):
    with open(file_path, "w") as f:
        json.dump(opt.__dict__, f, indent=2)

def cosine_similarity_matrix(a,b):
    if 'numpy' in str(type(a)):
        return cosine_similarity(a,b)
    else:
        return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)


