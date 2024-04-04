using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using Xunit;
using ApprovalTests;
using ApprovalTests.Reporters;
using ApprovalTests.Namers;
using TorchSharp;
using System.Text;
using System.Runtime.InteropServices;

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
    public async Task Fp16ShapeTest()
    {
        var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/text_encoder";
        var clipTextModel = CLIPTextModel.FromPretrained(modelWeightFolder, torchDtype: ScalarType.Float16);
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

    [Fact(Skip = "need cuda")]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task Fp16TextModelForwardTest()
    {
        // Comment out the following two line and install torchsharp-cuda package if your machine support Cuda 12
        var libTorch = "/home/xiaoyuz/diffusers/venv/lib/python3.8/site-packages/torch/lib/libtorch.so";
        NativeLibrary.Load(libTorch);
        var dtype = ScalarType.Float16;
        var device = DeviceType.CUDA;
        torch.InitializeDeviceType(device);
        var modelWeightFolder = "/home/xiaoyuz/stable-diffusion-2/text_encoder";
        var clipTextModel = CLIPTextModel.FromPretrained(modelWeightFolder, torchDtype: dtype);
        clipTextModel = clipTextModel.to(device);
        long[] input_ids = [49406,   320,  1125,   539,   320,  2368, 49407, 49406,   320,  1125,   539,   320,  1929, 49407]; // a photo of a cat a photo of a dog
        long[] attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

        var input_ids_tensor = input_ids.ToTensor([2, 7]).to(device);
        var attention_mask_tensor = attention_mask.ToTensor([2, 7]).to(device);
        input_ids_tensor.Peek("input_ids_tensor");
        attention_mask_tensor.Peek("attention_mask_tensor");
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