using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using Xunit;
using ApprovalTests;
using ApprovalTests.Reporters;
using ApprovalTests.Namers;
using TorchSharp;
using System.Text;

namespace SD;

public class StableDiffusionPipelineTest
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void GenerateCatImageTest()
    {
        var tokenzierModelPath = "/home/xiaoyuz/stable-diffusion-2/tokenizer";
        var unetModelPath = "/home/xiaoyuz/stable-diffusion-2/unet";
        var textModelPath = "/home/xiaoyuz/stable-diffusion-2/text_encoder";
        var schedulerModelPath = "/home/xiaoyuz/stable-diffusion-2/scheduler";
        var vaeModelPath = "/home/xiaoyuz/stable-diffusion-2/vae";

        var tokenizer = BPETokenizer.FromPretrained(tokenzierModelPath);
        var unet = UNet2DConditionModel.FromPretrained(unetModelPath, torchDtype: ScalarType.Float32);
        var clipTextModel = CLIPTextModel.FromPretrained(textModelPath, torchDtype: ScalarType.Float32);
        var ddim = DDIMScheduler.FromPretrained(schedulerModelPath);
        var vae = AutoencoderKL.FromPretrained(vaeModelPath);
        var dtype = ScalarType.Float32;
        var device = DeviceType.CPU;
        var generator = torch.manual_seed(0);
        var input = "a photo of a cat";
        var latent = torch.arange(0, 1 * 4 * 96 * 96);
        latent = latent.reshape(1, 4, 96, 96).to(dtype).to(device);
        var pipeline = new StableDiffusionPipeline(
            vae: vae,
            text_encoder: clipTextModel,
            unet: unet,
            tokenizer: tokenizer,
            scheduler: ddim);

        var images = pipeline.Run(
            prompt: input,
            num_inference_steps: 5,
            generator: generator,
            latents: latent);

        var sb = new StringBuilder();
        sb.Append(images.Images.Peek("images"));

        Approvals.Verify(sb.ToString());
    }
}