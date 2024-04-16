using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace SD;

public class UpDecoderBlock2D : Module<Tensor, Tensor?, Tensor>
{
    private readonly int in_channels;
    private readonly int out_channels;
    private readonly int? resolution_idx;
    private readonly float dropout;
    private readonly int num_layers;
    private readonly float resnet_eps;
    private readonly string resnet_time_scale_shift;
    private readonly string resnet_act_fn;
    private readonly int resnet_groups;
    private readonly bool resnet_pre_norm;
    private readonly float output_scale_factor;
    private readonly bool add_upsample;
    private readonly int? temb_channels;
    private readonly ScalarType dtype;

    private readonly ModuleList<Module<Tensor, Tensor?, Tensor>> resnets;
    private readonly ModuleList<Module<Tensor, long[]?, Tensor>>? upsamplers = null;
    public UpDecoderBlock2D(
        int in_channels,
        int out_channels,
        int? resolution_idx = null,
        float dropout = 0.0f,
        int num_layers = 1,
        float resnet_eps = 1e-6f,
        string resnet_time_scale_shift = "default",
        string resnet_act_fn = "swish",
        int resnet_groups = 32,
        bool resnet_pre_norm = true,
        float output_scale_factor = 1.0f,
        bool add_upsample = true,
        int? temb_channels = null,
        ScalarType dtype = ScalarType.Float32)
        : base(nameof(UpDecoderBlock2D))
    {
        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.resolution_idx = resolution_idx;
        this.dropout = dropout;
        this.num_layers = num_layers;
        this.resnet_eps = resnet_eps;
        this.resnet_time_scale_shift = resnet_time_scale_shift;
        this.resnet_act_fn = resnet_act_fn;
        this.resnet_groups = resnet_groups;
        this.resnet_pre_norm = resnet_pre_norm;
        this.output_scale_factor = output_scale_factor;
        this.add_upsample = add_upsample;
        this.temb_channels = temb_channels;
        this.dtype = dtype;

        this.resnets = new ModuleList<Module<Tensor, Tensor?, Tensor>>();
        for(int i = 0; i!= num_layers; ++i)
        {
            var input_channels = i == 0 ? in_channels : out_channels;
            if (resnet_time_scale_shift == "spatial")
            {
                resnets.Add(
                    new ResnetBlockCondNorm2D(
                        in_channels: input_channels,
                        out_channels: out_channels,
                        temb_channels: temb_channels ?? 512,
                        eps: resnet_eps,
                        groups: resnet_groups,
                        time_embedding_norm: "spatial",
                        non_linearity: resnet_act_fn,
                        output_scale_factor: output_scale_factor,
                        dtype: dtype)
                );
            }
            else
            {
                resnets.Add(
                    new ResnetBlock2D(
                        in_channels: input_channels,
                        out_channels: out_channels,
                        temb_channels: temb_channels,
                        groups: resnet_groups,
                        pre_norm: resnet_pre_norm,
                        eps: resnet_eps,
                        non_linearity: resnet_act_fn,
                        time_embedding_norm: resnet_time_scale_shift,
                        output_scale_factor: output_scale_factor,
                        dtype: dtype)
                );
            }
        }

        if (add_upsample)
        {
            this.upsamplers = new ModuleList<Module<Tensor, long[]?, Tensor>>();
            this.upsamplers.Add(new Upsample2D(
                channels: out_channels,
                use_conv: true,
                out_channels: out_channels,
                dtype: dtype
            ));
        }

        this.resolution_idx = resolution_idx;
    }

    public override Tensor forward(Tensor hidden_states, Tensor? temb)
    {
        foreach (var resnet in resnets)
        {
            hidden_states = resnet.forward(hidden_states, temb);
        }

        if (upsamplers != null)
        {
            foreach (var upsample in upsamplers)
            {
                hidden_states = upsample.forward(hidden_states, null);
            }
        }

        return hidden_states;
    }
}