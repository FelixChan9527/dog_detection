from tqdm import tqdm
from loss import *
from config import *
from model import *
from dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader

def train_fn(model, dataset, epoch, model_path):
    optimizer = optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(FEATURE_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(DEVICE)
    
    for i in range(epoch):
        train_loader = DataLoader(dataset, 16, shuffle=True)
        model.train()
        loop = tqdm(train_loader)
        losses = []
        for data in loop:
            imgs, labels = data
            y0, y1, y2 = (
                labels[0].to(DEVICE),
                labels[1].to(DEVICE),
                labels[2].to(DEVICE)
            )
            imgs = imgs.to(DEVICE)
            
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = (
                    loss_fn(out[0], y0, anchors[0])
                    + loss_fn(out[1], y1, anchors[1])
                    + loss_fn(out[2], y2, anchors[2])
                )
            
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss)
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    model = YoloNet(3, 120).to(DEVICE)
    dataset = YoloDataset(IMGS_PATH, ANNATATIONS_PATH, NAMES_FILE, 
                            IMG_SIZE, ANCHORS)
    train_fn(model, dataset, 50, MODEL_PATH)
