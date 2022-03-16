import torch

IMG_SIZE = 416
ANNATATIONS_PATH = "/home/MyServer/data/dogs/annotation/annotation"
IMGS_PATH = "/home/MyServer/data/dogs/images/images"
NAMES_FILE = "/home/MyServer/My_Code/MachineLearning/dog_detect/dogs_names"
MODEL_PATH = "/home/MyServer/My_Code/MachineLearning/dog_detect/dog_dect.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]
FEATURE_SIZE = [13, 26, 52]