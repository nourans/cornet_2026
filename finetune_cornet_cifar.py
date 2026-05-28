import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cornet import cornet_s

device = torch.device("cuda")
assert torch.cuda.is_available()

# ─── 1. LOAD MODEL ───────────────────────────────────────────────
model = cornet_s(pretrained=True)
model = model.module if hasattr(model, 'module') else model
model.decoder.linear = nn.Linear(512, 10)  # replace head

# ─── 2. FREEZE everything except the new head ────────────────────
# We don't want to update ImageNet weights yet — just train the new head first
for name, param in model.named_parameters():
    if "decoder.linear" not in name:
        param.requires_grad = False  # frozen
    else:
        param.requires_grad = True   # trainable

model.eval()  # for batchnorm/dropout
model.to(device)

# ─── 3. DATA ─────────────────────────────────────────────────────
# CORnet was pretrained on ImageNet (224×224, ImageNet stats)
# so we keep that preprocessing even when fine-tuning on CIFAR-10
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(224),                        # match ImageNet pretraining
    transforms.RandomHorizontalFlip(),             # light augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,  num_workers=4)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=64, shuffle=False, num_workers=4)

# ─── 4. TRAINING SETUP ───────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# Only pass parameters that require gradients to the optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

# ─── 5. TRAIN ────────────────────────────────────────────────────
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

NUM_EPOCHS = 10  # head-only training converges fast

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    acc = evaluate(model)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {running_loss/len(trainloader):.3f} | Test Acc: {acc:.2%}")

# ─── 6. OPTIONAL: UNFREEZE and fine-tune entire model ────────────
# Once the head is trained, unfreeze everything for a few more epochs
# Use a much smaller lr so you don't destroy the ImageNet features
print("\nUnfreezing all layers for full fine-tuning...")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 100x smaller lr

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    acc = evaluate(model)
    print(f"Full FT Epoch {epoch+1}/5 | Loss: {running_loss/len(trainloader):.3f} | Test Acc: {acc:.2%}")

# ─── 7. SAVE ─────────────────────────────────────────────────────
torch.save(model.state_dict(), "cornet_s_cifar10.pth")
print("Saved to cornet_s_cifar10.pth")