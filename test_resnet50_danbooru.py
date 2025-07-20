import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import urllib.request
from PIL import Image
import albumentations as A

# Загрузка модели RF5 через torch.hub
model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50', pretrained=True)
model.eval()

# Загрузка тегов из официального json файла RF5
url = "https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json"
with urllib.request.urlopen(url) as response:
    class_names = json.loads(response.read().decode())

# Предобработка изображения
transform = transforms.Compose([
    transforms.Resize((360, 360)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.713739812374115, 0.6627991795539856, 0.6518916487693787], std=[0.2969885468482971, 0.3017076551914215, 0.2979130446910858])
])

# Загружаем изображение
image_path = 'test_image.png'
input_image = Image.open(image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0)

# Инференс
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.sigmoid(output).squeeze(0)

# Визуализация картинки
plt.imshow(input_image)
plt.axis('off')

# Визуализация тегов
def plot_text(thresh=0.2):
    selected = probs[probs > thresh]
    inds = probs.argsort(descending=True)
    txt = 'Predictions (prob > {:.2f}):\n'.format(thresh)
    for i in inds[:len(selected)]:
        txt += f"{class_names[i]}: {probs[i].item():.3f}\n"
    plt.text(input_image.size[0]*1.05, input_image.size[1]*0.85, txt, fontsize=10)

plot_text(thresh=0.2)
plt.tight_layout()
plt.show()
