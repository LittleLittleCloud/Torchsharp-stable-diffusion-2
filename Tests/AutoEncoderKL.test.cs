using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using Xunit;
using ApprovalTests;
using ApprovalTests.Reporters;
using ApprovalTests.Namers;
using TorchSharp;

namespace SD;

public class AutoEncoderKLTest
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task ShapeTest()
    {
        var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/vae";
        var autoKL = AutoencoderKL.FromPretrained(modelWeightFolder, torchDtype: ScalarType.Float32);
        var state_dict_str = autoKL.Peek();
        Approvals.Verify(state_dict_str);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task Fp16ShapeTest()
    {
        var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/vae";
        var autoKL = AutoencoderKL.FromPretrained(modelWeightFolder, torchDtype: ScalarType.Float16);
        var state_dict_str = autoKL.Peek();
        Approvals.Verify(state_dict_str);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task EncoderForwardTest()
    {
        var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/vae";
        var autoKL = AutoencoderKL.FromPretrained(modelWeightFolder, torchDtype: ScalarType.Float32);
        var latent = torch.arange(0, 1 * 3 * 512 * 512, dtype: ScalarType.Float32);
        latent = latent.reshape(1, 3, 512, 512);

        var result = autoKL.Encoder.forward(latent);
        var str = result.Peek("autokl_encoder_forward");
        Approvals.Verify(str);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task DecoderForwardTest()
    {
        var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/vae";
        var autoKL = AutoencoderKL.FromPretrained(modelWeightFolder, torchDtype: ScalarType.Float32);
        var latent = torch.arange(0, 1 * 4 * 96 * 96, dtype: ScalarType.Float32);
        latent = latent.reshape(1, 4, 96, 96);

        var result = autoKL.Decoder.forward(latent);
        var str = result.Peek("autokl_decoder_forward");
        Approvals.Verify(str);
    }
}