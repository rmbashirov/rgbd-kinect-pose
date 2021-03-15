# face_expression

## How to use Inferer and Cropper
1. Clone `face_expression` repository:
```bash
git clone https://github.sec.samsung.net/k-iskakov/face_expression
```

2. Make sure you have required modules (`requirements.txt`).

3. Install `face_expression` package:
```bash
bash setup.sh
```
 
3. Inferer is instantiated with paths to model's `config` and `checkpoint`. Cropper requires kinect camera parameters. Latest checkpoint:
```python
checkpoint_path = "/Vol1/dbstore/datasets/k.iskakov/share/face_expression/gold_checkpoints/siamese+mouth+checkpoint_000044/checkpoint_000044.pth"
config_path = "/Vol1/dbstore/datasets/k.iskakov/share/face_expression/gold_checkpoints/siamese+mouth+checkpoint_000044/config.yaml"
```

4. Now you can instantiate Inferer and Cropper from any place on your machine. Example `notebooks/vis_azure_people_with_inferer.ipynb`.

