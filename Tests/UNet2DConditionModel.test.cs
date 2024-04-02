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
        var state_dict_str = unet.Peek_Shape();
        Approvals.Verify(state_dict_str);
    }
}