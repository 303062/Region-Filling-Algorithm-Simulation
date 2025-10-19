
# Region Filling Algorithm

A Python implementation of the region-filling algorithm described in [Object removal by exemplar-based inpainting](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_cvpr2003.pdf) by Criminisi et al.

---
## Dependencies

* `numpy`
* `scikit-image`
* `opencv-python`

---

## Usage

```bash
python3 main.py -i <input_image> -m <mask_image> -p <patch_size> -o <output_image>
```

### Parameters

| Argument             | Description                                | Required | Default    |
| -------------------- | ------------------------------------------ | -------- | ---------- |
| `-i`, `--img`        | Path to the input image                    | Yes      | -          |
| `-m`, `--mask`       | Path to the binary mask image              | Yes      | -          |
| `-p`, `--patch-size` | Square patch size (must be odd)            | No       | 9          |
| `-o`, `--output`     | Output path and filename                   | No       | result.png | 

You can also use the following command to show the arguments:

```bash
python3 main.py -h
```

**Note:**
- Ensure the mask is a binary image where white (255) indicates the region to fill.  
- The script will adjust the patch size if it is not odd or is too large for the image. 

---

## Examples

```bash
python3 main.py -i img/object3.png -m img/mask3.png -p 13 -o result3.png
```

The command will inpaint the `image3.png` using the `mask3.png` with a patch size of 13 and save the result as `result3.png`.