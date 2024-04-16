using System.Runtime.InteropServices;
using TorchSharp;
using SD;
var dtype = ScalarType.Float16;
var device = DeviceType.CUDA;
var outputFolder = "img";
torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

if (!Directory.Exists(outputFolder))
{
    Directory.CreateDirectory(outputFolder);
}

// Comment out the following two line and install torchsharp-cuda package if your machine support Cuda 12
// var libTorch = "/home/xiaoyuz/diffusers/venv/lib/python3.8/site-packages/torch/lib/libtorch.so";
// NativeLibrary.Load(libTorch);
torch.InitializeDeviceType(device);
if (!torch.cuda.is_available())
{
    device = DeviceType.CPU;
}

var input = "a photo of cat chasing after dog";
var modelFolder = @"C:\Users\xiaoyuz\source\repos\stable-diffusion-2\";
var pipeline = StableDiffusionPipeline.FromPretrained(modelFolder, torchDtype: dtype);
pipeline.To(device);

var output = pipeline.Run(
    prompt: input,
    width: 1020,
    height: 768,
    num_inference_steps: 50
    );

var decoded_images = torch.clamp((output.Images + 1.0) / 2.0, 0.0, 1.0);

for(int i = 0; i!= decoded_images.shape[0]; ++i)
{
    var savedPath = Path.Join(outputFolder, $"{i}.png");
    var image = decoded_images[i];
    image = (image * 255.0).to(torch.ScalarType.Byte).cpu();
    torchvision.io.write_image(image, savedPath, torchvision.ImageFormat.Png);

    Console.WriteLine($"save image to {savedPath}, enjoy");
}