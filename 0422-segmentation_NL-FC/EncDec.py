import torchvision

vgg_model = torchvision.models.vgg16(pretrained=True)

print(vgg_model)
