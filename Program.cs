using SD;
using static TorchSharp.torch;

var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/vae";
var autoKL = AutoencoderKL.FromPretrained(modelWeightFolder, torchDtype: ScalarType.Float32);
autoKL.Peek();