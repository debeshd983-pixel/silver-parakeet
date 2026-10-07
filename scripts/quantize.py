"""Dynamic INT8 quantization for ONNX models using onnxruntime.quantization."""
import argparse
import os
import onnxruntime.quantization as ort_quant


def quantize_model(input_path: str, output_path: str):
    print(f"Quantizing {input_path} -> {output_path} (INT8 dynamic)...")
    ort_quant.quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=ort_quant.QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=True,
        reduce_range=True,
    )
    in_size = os.path.getsize(input_path) / (1024 * 1024)
    out_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done: {in_size:.1f} MB -> {out_size:.1f} MB ({out_size/in_size*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX models to INT8")
    parser.add_argument("--models-dir", default="models", help="Directory containing ONNX models")
    args = parser.parse_args()

    for name in ["classifier.onnx", "clip_encoder.onnx"]:
        in_p = os.path.join(args.models_dir, name)
        if os.path.exists(in_p):
            out_p = os.path.join(args.models_dir, name.replace(".onnx", ".int8.onnx"))
            quantize_model(in_p, out_p)
        else:
            print(f"Model {in_p} not found; skipping quantization.")
