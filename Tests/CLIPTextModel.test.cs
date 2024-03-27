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

public class CLIPTextModelTest
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task ShapeTest()
    {
        var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/text_encoder";
        var clipTextModel = CLIPTextModel.FromPretrained(modelWeightFolder, torchDtype: ScalarType.Float32);
        var state_dict_str = clipTextModel.Peek();
        Approvals.Verify(state_dict_str);
    }

    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task TextModelForwardTest()
    {
        var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/text_encoder";
        var clipTextModel = CLIPTextModel.FromPretrained(modelWeightFolder, torchDtype: ScalarType.Float32);
        long[] input_ids = [49406,   320,  1125,   539,   320,  2368, 49407, 49406,   320,  1125,   539,   320,  1929, 49407]; // a photo of a cat a photo of a dog

        long[] attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        var input_ids_tensor = input_ids.ToTensor([2, 7]);
        var attention_mask_tensor = attention_mask.ToTensor([2, 7]);

        var result = clipTextModel.forward(input_ids_tensor, attention_mask_tensor);
        var last_hidden_state = result.LastHiddenState;
        var pooled_output = result.PoolerOutput;

        var last_hidden_state_str = last_hidden_state.Peek("clip_text_model_forward");
        var pooled_output_str = pooled_output.Peek("clip_text_model_forward");
        var sb = new StringBuilder();
        sb.AppendLine(last_hidden_state_str);
        sb.AppendLine(pooled_output_str);

        Approvals.Verify(sb.ToString());
    }
}