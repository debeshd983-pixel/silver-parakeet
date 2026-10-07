"""Exports PyTorch classifier and CLIP image encoder to ONNX with opset 17 and dynamic batch axis."""
import argparse
import hashlib
import os
import torch
from transformers import AutoModelForImageClassification, CLIPVisionModelWithProjection


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_classifier(model_id_or_path: str, output_path: str):
    print(f"Exporting classifier from {model_id_or_path} -> {output_path}...")
    model = AutoModelForImageClassification.from_pretrained(model_id_or_path)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print(f"Successfully exported classifier to {output_path} (SHA-256: {compute_sha256(output_path)[:12]})")


class CLIPImageEncoderWrapper(torch.nn.Module):
    """Wraps CLIP vision model to project embeddings."""
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        return outputs.image_embeds


def export_clip(model_id_or_path: str, output_path: str):
    print(f"Exporting CLIP image encoder from {model_id_or_path} -> {output_path}...")
    vision_model = CLIPVisionModelWithProjection.from_pretrained(model_id_or_path)
    wrapper = CLIPImageEncoderWrapper(vision_model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "image_embeds": {0: "batch_size"}},
    )
    print(f"Successfully exported CLIP encoder to {output_path} (SHA-256: {compute_sha256(output_path)[:12]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--classifier", default="Organika/sdxl-detector", help="HF classifier repo or local path")
    parser.add_argument("--clip", default="openai/clip-vit-base-patch16", help="HF CLIP repo or local path")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cls_out = os.path.join(args.output_dir, "classifier.onnx")
    clip_out = os.path.join(args.output_dir, "clip_encoder.onnx")

    export_classifier(args.classifier, cls_out)
    export_clip(args.clip, clip_out)

    # Write SHA-256 checksums
    checksum_file = os.path.join(args.output_dir, "checksums.sha256")
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write(f"{compute_sha256(cls_out)}  classifier.onnx\n")
        f.write(f"{compute_sha256(clip_out)}  clip_encoder.onnx\n")
    print(f"Wrote checksums to {checksum_file}")
