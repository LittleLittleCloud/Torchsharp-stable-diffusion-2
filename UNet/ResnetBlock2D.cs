using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using System.Linq.Expressions;

namespace SD;

public class ResnetBlock2D : Module<Tensor, Tensor?, Tensor>
{
    private readonly bool pre_norm;
    private readonly int in_channels;
    private readonly int out_channels;
    private readonly bool use_conv_shortcut;
    private readonly bool up;
    private readonly bool down;
    private readonly float output_scale_factor;
    private readonly string time_embedding_norm;
    private bool skip_time_act;
    private readonly bool use_in_shortcut;

    private Module<Tensor, Tensor> norm1;
    private Module<Tensor, Tensor> conv1;
    private Module<Tensor, Tensor> norm2;
    private Module<Tensor, Tensor> conv2;
    private Module<Tensor, Tensor> dropout;
    private Linear? time_emb_proj;
    private Module<Tensor, Tensor> nonlinearity;
    private Module<Tensor, int?, Tensor>? upsample = null;
    private Module<Tensor, Tensor>? downsample = null;
    private Module<Tensor, Tensor>? conv_shortcut = null;
    public ResnetBlock2D(
        int in_channels,
        int? out_channels = null,
        bool conv_shortcut = false,
        float dropout = 0.0f,
        int? temb_channels = 512,
        int groups = 32,
        int? groups_out = null,
        bool pre_norm = true,
        float eps = 1e-6f,
        string non_linearity = "swish",
        bool skip_time_act = false,
        string time_embedding_norm = "default", // default, scale_shift,
        Tensor? kernel = null,
        float output_scale_factor = 1.0f,
        bool?  use_in_shortcut = null,
        bool up = false,
        bool down = false,
        bool conv_shortcut_bias = true,
        int? conv_2d_out_channels = null)
        : base(nameof(ResnetBlock2D))
        {
            if (time_embedding_norm == "ada_group" || time_embedding_norm == "spatial")
            {
                throw new ArgumentException("Invalid time_embedding_norm: " + time_embedding_norm);
            }

            this.pre_norm = pre_norm;
            this.in_channels = in_channels;
            this.out_channels = out_channels ?? in_channels;
            this.use_conv_shortcut = conv_shortcut;
            this.up = up;
            this.down = down;
            this.output_scale_factor = output_scale_factor;
            this.time_embedding_norm = time_embedding_norm;
            this.skip_time_act = skip_time_act;

            groups_out = groups_out ?? groups;

            this.norm1 = nn.GroupNorm(num_groups: groups, num_channels: in_channels, eps: eps, affine: true);
            this.conv1 = nn.Conv2d(in_channels, this.out_channels, kernelSize: 3, stride: 1, padding: 1, bias: true);

            if (temb_channels is not null)
            {
                if (this.time_embedding_norm == "default"){
                    this.time_emb_proj = nn.Linear(temb_channels.Value, this.out_channels);
                }
                else if (this.time_embedding_norm == "scale_shift")
                {
                    this.time_emb_proj = nn.Linear(temb_channels.Value, this.out_channels * 2);
                }
                else{
                    throw new ArgumentException("Invalid time_embedding_norm: " + time_embedding_norm);
                }
            }
            else{
                this.time_emb_proj = null;
            }

            this.norm2 = nn.GroupNorm(num_groups: groups_out.Value, num_channels: this.out_channels, eps: eps, affine: true);
            this.dropout = nn.Dropout(dropout);
            conv_2d_out_channels = conv_2d_out_channels ?? this.out_channels;
            this.conv2 = nn.Conv2d(this.out_channels, conv_2d_out_channels.Value, kernelSize: 3, stride: 1, padding: 1, bias: true);
            this.nonlinearity = Utils.GetActivation(non_linearity);
            if (this.up){
                this.upsample = new Upsample2D(channels: in_channels, use_conv: false);
            }
            else if (this.down){
                this.downsample = new Downsample2D(channels: in_channels, use_conv: false, padding: 1, name: "op");
            }

            this.use_in_shortcut = use_in_shortcut ?? this.in_channels != conv_2d_out_channels;

            if (this.use_in_shortcut)
            {
                this.conv_shortcut = nn.Conv2d(in_channels, this.out_channels, kernelSize: 1, stride: 1, padding: TorchSharp.Padding.Valid, bias: conv_shortcut_bias);
            }
        }

    public override Tensor forward(Tensor input_tensor, Tensor? temb)
    {
        var hidden_states = input_tensor;
        hidden_states = this.norm1.forward(hidden_states);
        hidden_states = this.nonlinearity.forward(hidden_states);
        if (this.upsample is not null){
            input_tensor = this.upsample.forward(input_tensor, null);
            hidden_states = this.upsample.forward(hidden_states, null);
        }
        else if (this.downsample is not null){
            input_tensor = this.downsample.forward(input_tensor);
            hidden_states = this.downsample.forward(hidden_states);
        }
        hidden_states = this.conv1.forward(hidden_states);
        if (this.time_emb_proj is not null)
        {
            temb.Peek("temb");
            if (!this.skip_time_act){
                temb = this.nonlinearity.forward(temb!);
            }
            temb = this.time_emb_proj.forward(temb!);
            // temb = self.time_emb_proj(temb)[:, :, None, None]
            temb = temb.unsqueeze(2).unsqueeze(3);
        }

        if (this.time_embedding_norm == "default"){
            if (temb is not null){
                hidden_states = hidden_states + temb;
            }
            hidden_states = this.norm2.forward(hidden_states);
        }
        else if (this.time_embedding_norm == "scale_shift")
        {
            if (temb is null){
                throw new ArgumentException("Time embedding is None");
            }

            var chunks = temb.chunk(2, 1);
            var time_scale = chunks[0];
            var time_shift = chunks[1];
            hidden_states = this.norm2.forward(hidden_states);
            hidden_states = hidden_states * (1 + time_scale) + time_shift;
        }
        else
        {
            hidden_states = this.norm2.forward(hidden_states);
        }

        hidden_states = this.nonlinearity.forward(hidden_states);
        hidden_states = this.dropout.forward(hidden_states);
        hidden_states = this.conv2.forward(hidden_states);
        hidden_states.Peek("hidden_states");
        if (this.conv_shortcut is not null)
        {
            input_tensor = this.conv_shortcut.forward(input_tensor);
        }


        return (input_tensor + hidden_states) / this.output_scale_factor;
    }
}