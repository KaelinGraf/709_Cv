import os
import numpy as np
import argparse

def resize_templates(input_dir="templates", output_dir="templates_resized", target_width=15):
    """
    Resizes all .npy templates in input_dir to target_width.
    Pads with zeros if smaller, crops symmetrically if larger.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Run template_extractor.py first or ensure {input_dir}/ exists.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.startswith("template_") and f.endswith(".npy")]
    if not files:
        print(f"No templates found in {input_dir}/.")
        return
        
    print(f"Found {len(files)} templates. Resizing to {target_width}px...")
    
    save_count = 0
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        template = np.load(filepath)
        
        # Squeeze in case shape is e.g. (1, W) instead of (W,)
        template = np.squeeze(template)
        current_width = len(template)
        
        if current_width == target_width:
            resized = template
        elif current_width < target_width:
            # Need to pad
            diff = target_width - current_width
            pad_left = diff // 2
            pad_right = diff - pad_left
            resized = np.pad(template, (pad_left, pad_right), mode='constant', constant_values=0)
        else:
            # Need to crop
            diff = current_width - target_width
            crop_left = diff // 2
            crop_right = current_width - (diff - crop_left)
            resized = template[crop_left:crop_right]
            
        print(f"[{filename}] {current_width}px -> {target_width}px")
        
        # Save output
        out_filename = os.path.join(output_dir, f"template_{target_width}px_{save_count}.npy")
        while os.path.exists(out_filename):
            save_count += 1
            out_filename = os.path.join(output_dir, f"template_{target_width}px_{save_count}.npy")
            
        # Add back batch dimension to match original format (1, W)
        np.save(out_filename, resized[np.newaxis, :])
        save_count += 1

    print(f"\n[SUCCESS] Saved {len(files)} resized templates to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize existing templates cleanly.")
    parser.add_argument("--width", type=int, default=6, help="Target template width in pixels")
    parser.add_argument("--input", type=str, default="templates", help="Input directory")
    parser.add_argument("--output", type=str, default="templates_resized", help="Output directory")
    
    args = parser.parse_args()
    resize_templates(input_dir=args.input, output_dir=args.output, target_width=args.width)
