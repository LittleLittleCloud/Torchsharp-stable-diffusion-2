using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class DownEncoderBlock2D : Module<Tensor, Tensor>
{
    private readonly ScalarType dtype;
    private readonly ModuleList<Module<Tensor, Tensor?, Tensor>> resnets;
    private readonly ModuleList<Module<Tensor,  Tensor>>? downsamplers;
    public DownEncoderBlock2D(
        int in_channels,
        int out_channels,
        float dropout = 0.0f,
        int num_layers = 1,
        float resnet_eps = 1e-6f,
        string resnet_time_scale_shift = "default",
        string resnet_act_fun = "swish",
        int resnet_groups = 32,
        bool resnet_pre_norm = true,
        float output_scale_factor = 1.0f,
        bool add_downsample = true,
        int downsample_padding = 1,
        ScalarType dtype = ScalarType.Float32)
        : base(nameof(DownEncoderBlock2D))
        {
            this.dtype = dtype;
            this.resnets = new ModuleList<Module<Tensor, Tensor?, Tensor>>();
            for (int i = 0; i < num_layers; i++)
            {
                in_channels = i == 0 ? in_channels : out_channels;
                if (resnet_time_scale_shift == "spatial")
                {
                    this.resnets.Add(new ResnetBlockCondNorm2D(
                        in_channels: in_channels,
                        out_channels: out_channels,
                        dropout: dropout,
                        temb_channels: out_channels,
                        groups: resnet_groups,
                        eps: resnet_eps,
                        non_linearity: resnet_act_fun,
                        time_embedding_norm: resnet_time_scale_shift,
                        output_scale_factor: output_scale_factor,
                        up: false,
                        down: false,
                        conv_2d_out_channels: out_channels,
                        conv_shortcut: false,
                        conv_shortcut_bias: true,
                        dtype: dtype
                    ));
                }
                else
                {
                    this.resnets.Add(new ResnetBlock2D(
                        in_channels: in_channels,
                        out_channels: out_channels,
                        dropout: dropout,
                        temb_channels: null,
                        groups: resnet_groups,
                        pre_norm: resnet_pre_norm,
                        eps: resnet_eps,
                        non_linearity: resnet_act_fun,
                        time_embedding_norm: resnet_time_scale_shift,
                        output_scale_factor: output_scale_factor,
                        up: false,
                        down: false,
                        conv_2d_out_channels: out_channels,
                        conv_shortcut: false,
                        conv_shortcut_bias: true,
                        dtype: dtype
                    ));
                }
            }

            if (add_downsample)
            {
                this.downsamplers = new ModuleList<Module<Tensor, Tensor>>();
                this.downsamplers.Add(new Downsample2D(
                    channels: out_channels,
                    use_conv: true,
                    out_channels: out_channels,
                    padding: downsample_padding,
                    name: "op",
                    dtype: dtype
                ));
            }
        }

    public override Tensor forward(Tensor hidden_states)
    {
        for (int i = 0; i < resnets.Count; i++)
        {
            hidden_states = resnets[i].forward(hidden_states, null);
        }

        if (downsamplers is not null)
        {
            for (int i = 0; i < downsamplers.Count; i++)
            {
                hidden_states = downsamplers[i].forward(hidden_states);
            }
        }

        return hidden_states;
    }

}
