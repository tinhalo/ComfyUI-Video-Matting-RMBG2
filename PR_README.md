# Add BRIAAI RMBG-2.0 Support

This PR adds support for [BRIAAI RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) to the ComfyUI-Video-Matting plugin. RMBG-2.0 is the latest version of the background removal model from BRIAAI with improved performance.

## Features Added

- New `BRIAAI Matting V2` node for ComfyUI that uses the RMBG-2.0 model
- Direct integration with Hugging Face Transformers library
- Support for batch processing of video frames
- Maintains the same interface as the original RMBG node for ease of use

## Implementation Details

- Uses `AutoModelForImageSegmentation` from the transformers library
- Properly handles high precision matrix multiplication for improved performance
- Uses ImageNet normalization values [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
- Supports FP16 computation for faster processing and reduced memory usage

## Dependencies

Added dependency on the `transformers` library. Updated requirements.txt file.

## Testing

Tested with multiple video inputs at different resolutions, producing high quality alpha mattes.

## Screenshots

[Add screenshots of the node in action if available]
