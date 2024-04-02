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

public class DDIMSchedulerTest
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public async Task StepTest()
    {
        var dtype = ScalarType.Float32;
        var device = DeviceType.CPU;
        var path = "/home/xiaoyuz/stable-diffusion-2/scheduler";
        var ddim = DDIMScheduler.FromPretrained(path);
        var timestep = 1;
        ddim.SetTimesteps(timestep);

        var latent = torch.arange(0, 1 * 4 * 64 * 64);
        latent = latent.reshape(1, 4, 64, 64).to(dtype).to(device);

        var step_output = ddim.Step(latent, timestep, latent);
        var sb = new StringBuilder();
        sb.AppendLine(step_output.PrevSample.Peek("step_output.PrevSample"));
        sb.AppendLine(step_output.PredOriginalSample!.Peek("step_output.PredOriginalSample"));

        Approvals.Verify(sb.ToString());
    }
}