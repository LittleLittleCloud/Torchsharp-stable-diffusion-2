using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;


public class ResnetBlockCondNorm2D : Module<Tensor, Tensor?, Tensor>
{
    private readonly int in_channels;
    private readonly int out_channels;
    private readonly bool use_conv_shortcut;
    private readonly bool up;
    private readonly bool down;
    private readonly float output_scale_factor;
    private readonly string time_embedding_norm;
    private ScalarType defaultDtype;

    private readonly Module<Tensor, Tensor, Tensor> norm1;
    private readonly Module<Tensor, Tensor> conv1;
    private readonly Module<Tensor, Tensor, Tensor> norm2;
    private readonly Module<Tensor, Tensor> dropout;
    private readonly Module<Tensor, Tensor> conv2;
    private readonly Module<Tensor, Tensor> nonlinearity;
    private readonly Module<Tensor, int?, Tensor>? upsample;
    private readonly Module<Tensor, Tensor>? downsample;
    private readonly Module<Tensor, Tensor>? conv_shortcut;

    public ResnetBlockCondNorm2D(
        int in_channels,
        int? out_channels = null,
        bool conv_shortcut = false,
        float dropout = 0.0f,
        int temb_channels = 512,
        int groups = 32,
        int? groups_out = null,
        float eps = 1e-6f,
        string non_linearity = "swish",
        string time_embedding_norm = "ada_group",
        float output_scale_factor = 1.0f,
        bool? use_in_shortcut = null,
        bool up = false,
        bool down = false,
        bool conv_shortcut_bias = true,
        int? conv_2d_out_channels = null,
        ScalarType dtype = ScalarType.Float32)
        : base(nameof(ResnetBlockCondNorm2D))
    {
        this.in_channels = in_channels;
        this.out_channels = out_channels ?? in_channels;
        this.use_conv_shortcut = conv_shortcut;
        this.up = up;
        this.down = down;
        this.output_scale_factor = output_scale_factor;
        this.time_embedding_norm = time_embedding_norm;

        groups_out = groups_out ?? groups;

        if (this.time_embedding_norm == "ada_group")
        {
            this.norm1 = new AdaGroupNorm(
                embedding_dim: temb_channels,
                out_dim: this.in_channels,
                num_groups: groups,
                eps: eps,
                dtype: dtype);
            this.norm2 = new AdaGroupNorm(
                embedding_dim: temb_channels,
                out_dim: this.out_channels,
                num_groups: groups_out.Value,
                eps: eps,
                dtype: dtype);
        }
        else if (this.time_embedding_norm == "spatial")
        {
            this.norm1 = new SpatialNorm(
                f_channels: this.in_channels,
                zq_channels: temb_channels,
                dtype: dtype);
            this.norm2 = new SpatialNorm(
                f_channels: this.out_channels,
                zq_channels: temb_channels,
                dtype: dtype);
        }
        else
        {
            throw new ArgumentException("Invalid time_embedding_norm");
        }

        this.conv1 = nn.Conv2d(
            inputChannel: this.in_channels,
            outputChannel: this.in_channels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dtype: dtype);

        this.dropout = nn.Dropout(dropout);
        conv_2d_out_channels = conv_2d_out_channels ?? this.out_channels;
        this.conv2 = nn.Conv2d(
            inputChannel: this.in_channels,
            outputChannel: conv_2d_out_channels.Value,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            dtype: dtype);
        this.nonlinearity = Utils.GetActivation(non_linearity);

        this.upsample = null;
        this.downsample = null;
        if (this.up)
        {
            this.upsample = new Upsample2D(
                channels: this.in_channels,
                use_conv: false,
                dtype: dtype);
        }
        else if (this.down)
        {
            this.downsample = new Downsample2D(
                channels: this.in_channels,
                use_conv: false,
                padding: 1,
                name: "op",
                dtype: dtype);
        }

        this.use_conv_shortcut = use_in_shortcut ?? this.in_channels != conv_2d_out_channels;

        if (this.use_conv_shortcut)
        {
            this.conv_shortcut = nn.Conv2d(
                inputChannel: this.in_channels,
                outputChannel: conv_2d_out_channels.Value,
                kernelSize: 1,
                stride: 1,
                padding: Padding.Valid,
                bias: conv_shortcut_bias,
                dtype: dtype);
        }
    }

    public override Tensor forward(Tensor input_tensor, Tensor? temb)
    {
        var hidden_states = input_tensor;
        hidden_states = this.norm1.forward(hidden_states, temb);
        hidden_states = this.nonlinearity.forward(hidden_states);

        if (this.up)
        {
            input_tensor = this.upsample!.forward(input_tensor, null);
            hidden_states = this.upsample!.forward(hidden_states, null);
        }
        else if (this.down)
        {
            input_tensor = this.downsample!.forward(input_tensor);
            hidden_states = this.downsample!.forward(hidden_states);
        }

        hidden_states = this.conv1.forward(hidden_states);
        hidden_states = this.norm2.forward(hidden_states, temb);
        hidden_states = this.nonlinearity.forward(hidden_states);

        hidden_states = this.dropout.forward(hidden_states);
        hidden_states = this.conv2.forward(hidden_states);

        if (this.use_conv_shortcut)
        {
            input_tensor = this.conv_shortcut!.forward(input_tensor);
        }

        return (hidden_states + input_tensor) / this.output_scale_factor;
    }
}