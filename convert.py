import argparse
import torch
import onnxruntime
import numpy as np
from PIL import Image
import os
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

def main():
    
    # Download the model
    model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-base")
    processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-base")
    print("Downloaded the model")

    # Run inference on Huggingface model
    image = Image.open("~/Desktop/image.jpeg")
    inputs = processor(images=image, text="camera on the table", return_tensors="pt")

    # Placeholder for decoder input
    batch_size = inputs["flattened_patches"].shape[0]
    max_decoder_length = 50
    decoder_input_ids = torch.zeros((batch_size, max_decoder_length), dtype=torch.long)

    predictions = model.generate(flattened_patches=inputs["flattened_patches"],
                                attention_mask=inputs["attention_mask"],
                                decoder_input_ids=decoder_input_ids,
                                max_new_tokens=200)

    decoded_prediction = processor.decode(predictions[0], skip_special_tokens=True)
    print("Output")
    print(decoded_prediction)

    # Export the model to ONNX
    dummy_input = (inputs["flattened_patches"], inputs["attention_mask"], decoder_input_ids)
    onnx_model_path = "pix2struct.onnx"
    onnx_model_path = "model/pix2struct.onnx"
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
    torch.onnx.export(model, dummy_input, onnx_model_path,opset_version=11, verbose=True)

if __name__ == "__main__":
    main()
