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

public class UNet2DConditionModelTest
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task ShapeTest()
    {
        var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/unet";
        var unet = UNet2DConditionModel.FromPretrained(modelWeightFolder, torchDtype: ScalarType.Float32);
        var state_dict_str = unet.Peek();
        Approvals.Verify(state_dict_str);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task ForwardTest()
    {
        var dtype = ScalarType.Float32;
        var device = DeviceType.CPU;
        var textModelPath = "/home/xiaoyuz/stable-diffusion-2/text_encoder";
        var clipTextModel = CLIPTextModel.FromPretrained(textModelPath, torchDtype: ScalarType.Float32);

        var unetModelPath = "/home/xiaoyuz/stable-diffusion-2/unet";
        var unet = UNet2DConditionModel.FromPretrained(unetModelPath, torchDtype: ScalarType.Float32);

        var tokenizerPath = "/home/xiaoyuz/stable-diffusion-2/tokenizer";
        var tokenizer = BPETokenizer.FromPretrained(tokenizerPath);

        var latent = torch.arange(0, 1 * 4 * 96 * 96);
        latent = latent.reshape(1, 4, 96, 96).to(dtype).to(device);
        var input = "a photo of a cat";
        var text = tokenizer.Encode(input, true, true);
        var textTensor = torch.tensor(text, dtype: ScalarType.Int64).reshape(1, text.Length).to(device);
        var outputs = clipTextModel.forward(textTensor);
        var prompt_embeds = outputs.LastHiddenState;
        prompt_embeds.Peek("prompt_embeds");

        long[] t_candidates = [0L, 10L, 100L, 1000L, 2000L];
        var sb = new StringBuilder();
        sb.AppendLine(prompt_embeds.Peek("prompt_embeds"));

        foreach (var t_candidate in t_candidates)
        {
            var t = torch.tensor(t_candidate, dtype: ScalarType.Int64).to(device);
            var unetInput = new UNet2DConditionModelInput(latent, t, prompt_embeds);
            var output = unet.forward(unetInput);
            sb.AppendLine(output.Peek("output"));
        }
        
        Approvals.Verify(sb.ToString());
    }
}