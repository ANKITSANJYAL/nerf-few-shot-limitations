import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

# ----- Simple LoRA Wrapper -----
class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=4, alpha=16):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)

        # scale factor
        self.scaling = alpha / r

        # Initialize LoRA layers
        nn.init.kaiming_uniform_(self.lora_A.weight, a=0.01)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.scaling * self.lora_B(self.lora_A(x))

# ----- LoRADINO Model with Manual Injection -----
class LoRADINO(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", r=4, alpha=16):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Inject LoRA into attention: query, key, value
        for layer in self.backbone.encoder.layer:
            attn = layer.attention.attention
            attn.query = LoRALinear(attn.query, r=r, alpha=alpha)
            attn.key = LoRALinear(attn.key, r=r, alpha=alpha)
            attn.value = LoRALinear(attn.value, r=r, alpha=alpha)

    def forward(self, x):
        # x: raw PIL Image
        with torch.no_grad():
            processed = self.processor(images=x, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(next(self.backbone.parameters()).device)
        outputs = self.backbone(pixel_values=pixel_values)
        return outputs.last_hidden_state

