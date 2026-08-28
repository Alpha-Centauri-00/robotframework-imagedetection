import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


__version__ = "0.1.0"

IMG_SIZE = 200


class _CNN(nn.Module):
    def __init__(self, num_classes, img_size=IMG_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        # No global-average-pool here: it collapses each feature map to a single
        # number, discarding spatial layout. Classes that differ only by
        # orientation (e.g. a left- vs right-pointing arrow on an otherwise
        # identical sign) become indistinguishable after that collapse, so the
        # flatten below keeps the conv stack's spatial output intact instead.
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 3, img_size, img_size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class Base_Detect:

    def __init__(self):
        self.epoch = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def train_model(self, training_path, validation_path):
        print(f"Using device: {self.device}")

        train_dataset = ImageFolder(training_path, transform=self.transform)
        valid_dataset = ImageFolder(validation_path, transform=self.transform)

        train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=False)

        classes = train_dataset.classes
        model = _CNN(len(classes), IMG_SIZE).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(self.epoch):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            valid_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    valid_loss += criterion(outputs, labels).item() * images.size(0)
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)

            avg_valid_loss = valid_loss / total if total else 0.0
            accuracy = correct / total if total else 0.0
            print(f"Epoch {epoch + 1}/{self.epoch} - "
                  f"val_loss: {avg_valid_loss:.4f} - val_accuracy: {accuracy:.4f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "img_size": IMG_SIZE,
        }

        # annotate.py's --camera mode writes last_roi.json next to training/ and
        # validation/ (i.e. at the Data root) after a live-capture session. Pick
        # it up automatically so the crop region a tester calibrated travels
        # with the model instead of only living in a terminal printout or a
        # hand-copied .robot variable that's easy to forget to update.
        roi_file = Path(training_path).parent / "last_roi.json"
        if roi_file.is_file():
            try:
                capture_roi = json.loads(roi_file.read_text())
                if all(k in capture_roi for k in ("x", "y", "width", "height")):
                    checkpoint["capture_roi"] = capture_roi
                    print(f"Embedded capture ROI from {roi_file} into the checkpoint: {capture_roi}")
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Could not read {roi_file}, skipping capture_roi ({exc})")

        model_path = datetime.now().strftime("model_%Y%d%m%H%M.pt")
        torch.save(checkpoint, model_path)
        return model_path

    def get_capture_roi(self, model_name):
        """
        Returns the {"x", "y", "width", "height"} dict embedded in this
        checkpoint by train_model() (via annotate.py's last_roi.json), or
        None if this model has no stored ROI - e.g. it was trained before
        this feature existed, or without ever running annotate.py --camera.
        """
        checkpoint = torch.load(model_name, map_location=self.device)
        return checkpoint.get("capture_roi")

    def _predict(self, model_name, photo_path):
        checkpoint = torch.load(model_name, map_location=self.device)
        classes = checkpoint["classes"]
        img_size = checkpoint.get("img_size", IMG_SIZE)

        model = _CNN(len(classes), img_size).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        img = Image.open(photo_path).convert("RGB")
        X = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(X)
            predicted_index = outputs.argmax(dim=1).item()

        return classes[predicted_index]