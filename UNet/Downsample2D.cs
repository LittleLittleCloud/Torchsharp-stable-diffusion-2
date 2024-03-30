using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace SD;

public class Downsample2D : Module<Tensor, Tensor>
{
    private readonly int channels;
    private readonly int out_channels;
    private readonly bool use_conv;
    private readonly int? padding;
    private readonly string name;

    private readonly Module<Tensor, Tensor>? conv;
    private readonly Module<Tensor, Tensor>? Conv2d_0;
    private readonly Module<Tensor, Tensor>? norm;
    public Downsample2D(
        int channels,
        bool use_conv = false,
        int? out_channels = null,
        int? padding = 1,
        string name = "conv",
        int kernel_size = 3,
        string? norm_type = null,
        float eps = 1e-5f,
        bool elementwise_affine = false,
        bool bias = true)
        : base(nameof(Downsample2D))
        {
            this.channels = channels;
            this.out_channels = out_channels ?? channels;
            this.use_conv = use_conv;
            this.padding = padding;
            this.name = name;
            
            if (norm_type is "ln_norm")
            {
                this.norm = nn.LayerNorm(normalized_shape: this.channels, eps: eps, elementwise_affine: elementwise_affine);
            }
            else if (norm_type is null)
            {
                this.norm = null;
            }
            else
            {
                throw new ArgumentException("Invalid norm type: " + norm_type);
            }

            Module<Tensor, Tensor> conv;
            if (use_conv)
            {
                conv = nn.Conv2d(inputChannel: this.channels, outputChannel: this.out_channels, kernelSize: kernel_size, stride: 2, padding: padding, bias: bias);
            }
            else
            {
                var stride = 2;
                conv = nn.AvgPool2d(kernel_size: 2, stride: stride);
            }

            if (name == "conv"){
                this.Conv2d_0 = conv;
                this.conv = conv;
            }
            else if (name == "Conv2d_0"){
                this.Conv2d_0 = conv;
            }
            else
            {
                this.conv = conv;
            }

            RegisterComponents();
        }

    public override Tensor forward(Tensor hidden_states)
    {
        if (this.norm is not null)
        {
            hidden_states = this.norm.forward(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2);
        }

        if (this.use_conv && this.padding == 0)
        {
            hidden_states = nn.functional.pad(hidden_states, pad: [0, 1, 0, 1], mode: TorchSharp.PaddingModes.Constant, value: 0);
        }

        hidden_states = this.conv!.forward(hidden_states);

        return hidden_states;
    }
}
