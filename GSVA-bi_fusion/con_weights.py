import torch

sd = torch.load("/gemini/space/zhaozy/libingyu/GSVA-main/pretrained_weights/gsva_7b/gsva-7b-pt.bin", map_location="cpu", weights_only=True)
print(len(sd))
for k in list(sd.keys()):
    print(k)