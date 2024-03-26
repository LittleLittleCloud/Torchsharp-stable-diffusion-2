using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace SD;

public class UNetMidBlock2D : Module<Tensor, Tensor?, Tensor>
{
    // resnet_eps: float = 1e-6,
    //     resnet_time_scale_shift: str = "default",  # default, spatial
    //     resnet_act_fn: str = "swish",
    //     resnet_groups: int = 32,
    //     attn_groups: Optional[int] = None,
    //     resnet_pre_norm: bool = True,
    //     add_attention: bool = True,
    //     attention_head_dim: int = 1,
    //     output_scale_factor: float = 1.0,
    private readonly ModuleList<CrossAttention?> attentions;
    private readonly ModuleList<Module<Tensor, Tensor?, Tensor>> resnets;
    private readonly bool add_attention;
    public UNetMidBlock2D(
        int in_channels,
        int temb_channels = 512,
        float dropout = 0.0f,
        int num_layers = 1,
        float resnet_eps = 1e-6f,
        string resnet_time_scale_shift = "default",
        string resnet_act_fn = "swish",
        int? resnet_groups = 32,
        int? attn_groups = null,
        bool resnet_pre_norm = true,
        bool add_attention = true,
        int attention_head_dim = 1,
        float output_scale_factor = 1.0f)
        : base(nameof(UNetMidBlock2D))
    {
        resnet_groups = resnet_groups ?? Math.Min(in_channels / 4, 32);
        this.add_attention = add_attention;

        if (attn_groups is null)
        {
            attn_groups = resnet_time_scale_shift == "default" ? resnet_groups : null;
        }

        this.resnets = new ModuleList<Module<Tensor, Tensor?, Tensor>>();
        if (resnet_time_scale_shift == "spatial")
        {
            resnets.Add(
                new ResnetBlockCondNorm2D(
                    in_channels: in_channels,
                    out_channels: in_channels,
                    temb_channels: temb_channels,
                    eps: resnet_eps,
                    groups: resnet_groups ?? 32,
                    dropout: dropout,
                    time_embedding_norm: "spatial",
                    non_linearity: resnet_act_fn,
                    output_scale_factor: output_scale_factor)
            );
        }
        else
        {
            resnets.Add(
                new ResnetBlock2D(
                    in_channels: in_channels,
                    out_channels: in_channels,
                    temb_channels: temb_channels,
                    eps: resnet_eps,
                    groups: resnet_groups ?? 32,
                    dropout: dropout,
                    time_embedding_norm: resnet_time_scale_shift,
                    non_linearity: resnet_act_fn,
                    output_scale_factor: output_scale_factor,
                    pre_norm: resnet_pre_norm)
            );
        }

        var attentions = new ModuleList<CrossAttention?>();
        for(int i = 0; i!= num_layers; ++i)
        {
            if (add_attention)
            {
                attentions.Add(
                    new CrossAttention(
                        query_dim: in_channels,
                        heads: in_channels / attention_head_dim,
                        dim_head: attention_head_dim,
                        rescale_output_factor: output_scale_factor,
                        eps: resnet_eps,
                        norm_num_groups: attn_groups,
                        spatial_norm_dim: resnet_time_scale_shift == "spatial" ? temb_channels : null,
                        residual_connection: true,
                        bias: true,
                        upcast_softmax: true)
                );
            }
            else
            {
                attentions.Add(null);
            }

            if (resnet_time_scale_shift == "spatial")
            {
                resnets.Add(
                    new ResnetBlockCondNorm2D(
                        in_channels: in_channels,
                        out_channels: in_channels,
                        temb_channels: temb_channels,
                        eps: resnet_eps,
                        groups: resnet_groups ?? 32,
                        dropout: dropout,
                        time_embedding_norm: "spatial",
                        non_linearity: resnet_act_fn,
                        output_scale_factor: output_scale_factor)
                );
            }
            else
            {
                resnets.Add(
                    new ResnetBlock2D(
                        in_channels: in_channels,
                        out_channels: in_channels,
                        temb_channels: temb_channels,
                        eps: resnet_eps,
                        groups: resnet_groups ?? 32,
                        dropout: dropout,
                        time_embedding_norm: resnet_time_scale_shift,
                        non_linearity: resnet_act_fn,
                        output_scale_factor: output_scale_factor,
                        pre_norm: resnet_pre_norm)
                );
            }
        }

        this.attentions = attentions;
    }

    public override Tensor forward(Tensor hidden_states, Tensor? temb = null)
    {
        hidden_states = resnets[0].forward(hidden_states, temb);
        foreach (var (attn, resnet) in Enumerable.Zip(attentions, resnets.Skip(1)))
        {
            if (attn is not null)
            {
                hidden_states = attn.forward(hidden_states, temb: temb);
            }
            hidden_states = resnet.forward(hidden_states, temb);
        }
        
        return hidden_states;
    }
}