using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace SD;

public class SpatialNorm : Module<Tensor, Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> norm_layer;
    private readonly Module<Tensor, Tensor> conv_y;
    private readonly Module<Tensor, Tensor> conv_b;
    public SpatialNorm(
        int f_channels,
        int zq_channels
    ) : base(nameof(SpatialNorm))
    {
        this.norm_layer = nn.GroupNorm(num_channels: f_channels, num_groups: 32, eps: 1e-6f, affine: true);
        this.conv_y = nn.Conv2d(inputChannel: zq_channels, outputChannel: f_channels, kernelSize: 1, stride: 1, padding: TorchSharp.Padding.Valid);
        this.conv_b = nn.Conv2d(inputChannel: zq_channels, outputChannel: f_channels, kernelSize: 1, stride: 1, padding: TorchSharp.Padding.Valid);

        RegisterComponents();
    }

    public override Tensor forward(Tensor f, Tensor zq)
    {
        var f_size = f.shape[-2..];
        zq = nn.functional.interpolate(zq, f_size, mode: InterpolationMode.Nearest);
        var norm_f = this.norm_layer.forward(f);
        var new_f = norm_f * this.conv_y.forward(zq) + this.conv_b.forward(zq);

        return new_f;
    }
}