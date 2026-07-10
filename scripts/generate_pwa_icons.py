import os
import math
from PIL import Image, ImageDraw

def create_medical_icon(size):
    # Create clean white background
    img = Image.new('RGB', (size, size), color='#ffffff')
    draw = ImageDraw.Draw(img)
    
    center = size // 2
    r_outer = int(size * 0.35)
    r_inner = int(size * 0.12)
    
    # Draw radial iris lines
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        x_start = center + int(r_inner * 1.4 * math.cos(rad))
        y_start = center + int(r_inner * 1.4 * math.sin(rad))
        x_end = center + int(r_outer * 0.95 * math.cos(rad))
        y_end = center + int(r_outer * 0.95 * math.sin(rad))
        draw.line([x_start, y_start, x_end, y_end], fill='#e2e8f0', width=max(1, int(size * 0.015)))
        
    # Outer Limbus Circle (Purple)
    draw.arc(
        [center - r_outer, center - r_outer, center + r_outer, center + r_outer],
        start=0, end=360, fill='#a855f7', width=max(2, int(size * 0.035))
    )
    
    # Inner Pupil Circle (Cyan)
    draw.ellipse(
        [center - r_inner, center - r_inner, center + r_inner, center + r_inner],
        fill='#0ea5e9'
    )
    
    # Overlay a medical cross inside the pupil
    cross_w = max(1, int(r_inner * 0.3))
    cross_l = max(3, int(r_inner * 0.8))
    # Horizontal line
    draw.line([center - cross_l, center, center + cross_l, center], fill='#ffffff', width=cross_w)
    # Vertical line
    draw.line([center, center - cross_l, center, center + cross_l], fill='#ffffff', width=cross_w)
        
    # Save image
    return img

if __name__ == '__main__':
    public_dir = r"C:\Users\Bashar\Iris-Diabetes\frontend\public"
    os.makedirs(public_dir, exist_ok=True)
    
    # Generate 192x192
    icon_192 = create_medical_icon(192)
    icon_192.save(os.path.join(public_dir, "icon-192.png"))
    print("Generated icon-192.png")
    
    # Generate 512x512
    icon_512 = create_medical_icon(512)
    icon_512.save(os.path.join(public_dir, "icon-512.png"))
    print("Generated icon-512.png")
