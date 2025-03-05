import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import torch.nn.functional as F

from einops import rearrange
from comfy.model_management import soft_empty_cache, get_torch_device

from comfyui_vidmatt.utils import prepare_frames_color

# Model information
RMBG_MODEL_ID = "briaai/RMBG-2.0"
MODEL_INPUT_SIZE = (1024, 1024)
device = get_torch_device()

class BriaaiRembgV2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "fp16": ("BOOLEAN", {"default": True}),
                "bg_color": ("STRING", {"default": "green"}),
                "batch_size": ("INT", {"min": 1, "max": 64, "default": 4})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "matting"
    CATEGORY = "Video Matting"

    def matting(self, video_frames, fp16, bg_color, batch_size, **kwargs):
        # Load the model from Hugging Face
        model = AutoModelForImageSegmentation.from_pretrained(RMBG_MODEL_ID, trust_remote_code=True)
        
        # Set precision
        torch.set_float32_matmul_precision('high')
        
        # Prepare model
        if fp16:
            model = model.half()
        model = model.to(device)
        model.eval()

        # Prepare frames
        video_frames, orig_num_frames, bg_color = prepare_frames_color(video_frames, bg_color, batch_size)
        bg_color = bg_color.to(device)
        
        # Set image transformation
        orig_frame_size = video_frames.shape[2:4]
        transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        fgrs, masks = [], []
        for i in range(video_frames.shape[0] // batch_size):
            batch_imgs = video_frames[i*batch_size:(i+1)*batch_size].to(device)
            if fp16:
                batch_imgs = batch_imgs.half()
                
            # Resize and normalize input using the constant
            resized_input = F.interpolate(batch_imgs, size=MODEL_INPUT_SIZE, mode='bilinear')
            resized_input = transform(resized_input)
            
            # Get predictions
            with torch.no_grad():
                preds = model(resized_input)[-1].sigmoid()
                
            # Process masks
            mask = preds
            mask = F.interpolate(mask, size=orig_frame_size, mode='bilinear')

            fgr = batch_imgs * mask + bg_color * (1 - mask)
            fgrs.append(fgr.cpu())
            masks.append(mask.cpu().to(fgr.dtype))
            soft_empty_cache()
        
        fgrs = rearrange(torch.cat(fgrs), "n c h w -> n h w c")[:orig_num_frames].float().detach()
        masks = torch.cat(masks)[:orig_num_frames].squeeze(1).float().detach()
        return (fgrs, masks)
