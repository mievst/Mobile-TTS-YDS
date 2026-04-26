from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = Path("/Users/ruaoccj/Qwen3-TTS-Android/models/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX/vocoder.onnx")
output_model = Path("/Users/ruaoccj/Qwen3-TTS-Android/scripts/quantized/int8/vocoder_quantized_int8.onnx")

quantize_dynamic(
    model_input=str(input_model),
    model_output=str(output_model),
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
    per_channel=False,
    reduce_range=False,
)

print(f"Saved: {output_model}")