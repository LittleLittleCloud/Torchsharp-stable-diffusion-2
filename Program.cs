using System.Runtime.InteropServices;
using TorchSharp;
using SD;
var dtype = ScalarType.Float32;
var device = DeviceType.CUDA;
var outputFolder = "img";
torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

if (!Directory.Exists(outputFolder))
{
    Directory.CreateDirectory(outputFolder);
}

// Comment out the following two line if your machine support Cuda 12
var libTorch = "/home/xiaoyuz/diffusers/venv/lib/python3.8/site-packages/torch/lib/libtorch.so";
NativeLibrary.Load(libTorch);
torch.InitializeDeviceType(device);
if (!torch.cuda.is_available())
{
    device = DeviceType.CPU;
}
torch.set_default_dtype(dtype);

var tokenzierModelPath = "/home/xiaoyuz/stable-diffusion-2/tokenizer";
var unetModelPath = "/home/xiaoyuz/stable-diffusion-2/unet";
var textModelPath = "/home/xiaoyuz/stable-diffusion-2/text_encoder";
var schedulerModelPath = "/home/xiaoyuz/stable-diffusion-2/scheduler";
var vaeModelPath = "/home/xiaoyuz/stable-diffusion-2/vae";
var tokenizer = BPETokenizer.FromPretrained(tokenzierModelPath);
var clipTextModel = CLIPTextModel.FromPretrained(textModelPath, torchDtype: dtype);
var unet = UNet2DConditionModel.FromPretrained(unetModelPath, torchDtype: dtype);
var vae = AutoencoderKL.FromPretrained(vaeModelPath, torchDtype: dtype);
var ddim = DDIMScheduler.FromPretrained(schedulerModelPath);

var generator = torch.manual_seed(0);
var input = "a photo of an astronaut riding a horse on mars";
var latent = torch.arange(0, 1 * 4 * 96 * 96);
latent = latent.reshape(1, 4, 96, 96).to(dtype).to(device);
var pipeline = new StableDiffusionPipeline(
    vae: vae,
    text_encoder: clipTextModel,
    unet: unet,
    tokenizer: tokenizer,
    scheduler: ddim);

pipeline.To(device);

var output = pipeline.Run(
    prompt: input,
    num_inference_steps: 50,
    generator: generator);

output.Images.Peek("images");
var decoded_images = torch.clamp((output.Images + 1.0) / 2.0, 0.0, 1.0);

for(int i = 0; i!= decoded_images.shape[0]; ++i)
{
    var savedPath = Path.Join(outputFolder, $"{i}.png");
    var image = decoded_images[i];
    image = (image * 255.0).to(torch.ScalarType.Byte).cpu();
    torchvision.io.write_image(image, savedPath, torchvision.ImageFormat.Png);

    Console.WriteLine($"save image to {savedPath}, enjoy");
}