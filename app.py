import torch
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Import SAM 3 (after installing the repo)
#   git clone https://github.com/facebookresearch/sam3.git
#   cd sam3 && pip install -e .
from sam3.modeling.sam3 import build_sam3_model
from sam3.data.transforms import SAM3ImageProcessor


# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/sam3_base.pt"   # download from the SAM3 GitHub release page
TEXT_PROMPT = "person"                         # change to whatever concept you want
# ------------------------


def segment_image(image_path: str,
                  output_path: str = "segmented.png",
                  text_prompt: str = TEXT_PROMPT):
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Preprocess
    processor = SAM3ImageProcessor()
    batch = processor([img])  # returns dict with pixel_values etc.
    pixel_values = batch["pixel_values"].to(DEVICE)

    # Build model
    model = build_sam3_model(checkpoint=CHECKPOINT_PATH).to(DEVICE)
    model.eval()

    # Encode text prompt
    text_inputs = model.encode_text([text_prompt])  # returns embeddings + attention mask

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            text_embeds=text_inputs["text_embeds"],
            attention_mask=text_inputs["attention_mask"],
        )

    # Post-process instance segmentation
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=batch["original_sizes"].tolist()
    )[0]

    masks = results["masks"]  # [N, H, W] boolean
    scores = results["scores"]

    print(f"Found {len(masks)} objects for prompt '{text_prompt}'")

    # Overlay masks on image
    img_np = np.array(img).astype(np.float32) / 255.0
    overlay = img_np.copy()

    colors = plt.cm.get_cmap("tab20", len(masks))

    for i, mask in enumerate(masks):
        color = np.array(colors(i)[:3])
        m = mask[..., None]  # (H, W, 1)
        overlay = np.where(m, 0.5 * overlay + 0.5 * color, overlay)

    # Blend original and overlay
    blended = (0.5 * img_np + 0.5 * overlay)
    blended = (blended * 255).astype(np.uint8)

    out_img = Image.fromarray(blended)
    out_img.save(output_path)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="segmented.png")
    parser.add_argument("--prompt", type=str, default=TEXT_PROMPT)
    args = parser.parse_args()

    segment_image(args.image_path, args.output, args.prompt)
