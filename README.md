## Torchsharp Stable Diffusion 2

This repo contains a torchsharp implementation for [stable diffusion 2 model](https://github.com/Stability-AI/stablediffusion).

## Quick Start
To run the stable diffusion 2 model on your local machine, the following prerequisites are required:
- dotnet 6
- git lfs, this is to download the model file from hugging face

### Step 1: Get the model weight from huggingface
To get stable-diffusion-2 model weight, run the following command to download model weight from huggingface. Be sure to have git lfs installed.
```bash
git clone https://huggingface.co/stabilityai/stable-diffusion-2
```
> [!Note]
> To load fp32 model weight into GPU, it's recommended to have at least 16GB of GPU memory if you want to generate 768 * 768 size image. Loading fp16 model weight requires around 8GB of GPU memory.

### Step 2: Run the model
Clone this repo and replace the `modelFolder` folder with where you download huggingface model weight in [Program.cs](./Program.cs#L25)

Then run the following command to start the model:
```bash
dotnet run
```

### Example output
![a photo of an astronaut riding a horse on mars](./img/a%20photo%20of%20an%20astronaut%20riding%20a%20horse%20on%20mars.png)
(a photo of an astronaut riding a horse on mars)

### Load fp16 model weight for faster and more GPU memory efficient inference
You can load fp16 model weight by setting `dtype` to ` ScalarType.Float16` in [Program.cs](./Program.cs#L4). The inference on fp16 model weight is faster and more GPU memory efficient.

> [!Note]
> fp16 model only work with GPU because some operators doesn't work with fp16 and cpu.

### Update log
#### Update on 2024/04/03
- Add support for loading fp16 model weight
### See also
- [Torchsharp-llama](https://github.com/LittleLittleCloud/Torchsharp-llama): A torchsharp implementation for llama 2 model
- [Torchsharp-phi](https://github.com/LittleLittleCloud/Torchsharp-phi): A torchsharp implementation for phi model
