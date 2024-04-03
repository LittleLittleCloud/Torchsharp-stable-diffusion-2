using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace SD;

public class Upsample2D : Module<Tensor, int?, Tensor>
{
    private readonly int channels;
    private readonly bool use_conv;
    private readonly int out_channels;
    private readonly bool use_conv_transpose;
    private readonly string conv_name;
    private readonly bool interpolate;

    private readonly Module<Tensor, Tensor>? norm;
    private readonly Module<Tensor, Tensor>? conv;

    private readonly Module<Tensor, Tensor>? Conv2d_0;
    public Upsample2D(
        int channels,
        bool use_conv = false,
        bool use_conv_transpose = false,
        int? out_channels = null,
        string name = "conv",
        int? kernel_size = null,
        int padding = 1,
        string norm_type = null,
        float? eps = null,
        bool? elementwise_affine = null,
        bool bias = true,
        bool interpolate = true)
        : base(nameof(Upsample2D))
        {
            this.channels = channels;
            this.out_channels = out_channels ?? channels;
            this.use_conv = use_conv;
            this.use_conv_transpose = use_conv_transpose;
            this.conv_name = name;
            this.interpolate = interpolate;
            
            if (norm_type is "ln_norm")
            {
                this.norm = nn.LayerNorm(normalized_shape: this.channels, eps: eps ?? 1e-5, elementwise_affine: elementwise_affine?? true);
            }
            else if (norm_type is null)
            {
                this.norm = null;
            }
            else
            {
                throw new ArgumentException("Invalid norm type: " + norm_type);
            }

            Module<Tensor, Tensor>? conv;
            if (use_conv_transpose)
            {
                conv = nn.ConvTranspose2d(inputChannel: this.channels, outputChannel: this.out_channels, kernelSize: kernel_size ?? 4, stride: 2, padding: padding, bias: bias);
            }
            else if (use_conv)
            {
                conv = nn.Conv2d(inputChannel: this.channels, outputChannel: this.out_channels, kernelSize: kernel_size ?? 3, stride: 1, padding: padding, bias: bias);
            }
            else
            {
                conv = null;
            }

            if (this.conv_name is "conv")
            {
                this.conv = conv ?? throw new ArgumentException("Invalid conv type: " + this.conv_name);
            }
            else
            {
                this.Conv2d_0 = conv ?? throw new ArgumentException("Invalid conv type: " + this.conv_name);
            }

        }

    public override Tensor forward(Tensor hidden_states, int? output_size)
    {
        if (this.norm != null)
        {
            hidden_states = this.norm.forward(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2);
        }

        if (this.use_conv_transpose)
        {
            return this.conv!.forward(hidden_states);
        }

        var dtype = hidden_states.dtype;
        if (dtype == ScalarType.BFloat16){
            hidden_states = hidden_states.to_type(ScalarType.Float32);
        }

        if (hidden_states.shape[0] >= 64){
            hidden_states = hidden_states.contiguous();
        }

        if (this.interpolate){
            if (output_size is null){
                hidden_states = nn.functional.interpolate(hidden_states, scale_factor: [2, 2], mode: InterpolationMode.Nearest);
            }
            else{
                hidden_states = nn.functional.interpolate(hidden_states, size: [output_size.Value], mode: InterpolationMode.Nearest);
            }
        }

        if (dtype == ScalarType.BFloat16){
            hidden_states = hidden_states.to_type(ScalarType.BFloat16);
        }

        if (this.use_conv)
        {
            if (this.conv_name is "conv")
            {
                return this.conv!.forward(hidden_states);
            }
            else
            {
                return this.Conv2d_0!.forward(hidden_states);
            }
        }

        return hidden_states;
    }
}