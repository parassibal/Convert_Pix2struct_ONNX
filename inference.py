import argparse
import onnxruntime
import time
from PIL import Image
import os
from transformers import Pix2StructProcessor

def main():
    parser = argparse.ArgumentParser(description="Run inference on ONNX model.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the ONNX model folder")
    parser.add_argument("-i", "--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    onnx_model_path = os.path.join(args.model_path, "pix2struct.onnx")  # Construct the ONNX model path

    # Load the ONNX model
    onnx_session = onnxruntime.InferenceSession(onnx_model_path)

    # Load and preprocess the image
    image = Image.open(args.image_path)
    processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
    inputs = processor(images=image, text="camera on the table", return_tensors="pt")

    # Prepare input tensor for the ONNX model
    flattened_patches = inputs["flattened_patches"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    decoder_input_ids = inputs["decoder_input_ids"].numpy()

    # Run inference on the ONNX model
    start_time = time.time()
    outputs = onnx_session.run(None, {
        'flattened_patches': flattened_patches,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids
    })
    end_time = time.time()

    # Post-process the outputs
    decoded_prediction = processor.decode(outputs[0][0], skip_special_tokens=True)
    print("Output:")
    print(decoded_prediction)

    # Calculate inference time
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.4f} seconds")

if __name__ == "__main__":
    main()
