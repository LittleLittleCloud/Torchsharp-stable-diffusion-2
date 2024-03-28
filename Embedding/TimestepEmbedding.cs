using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class TimestepEmbedding : Module<Tensor, Tensor?, Tensor>
{
    private readonly Linear linear_1;
    private readonly Linear linear_2;
    private readonly Linear? cond_proj = null;
    private readonly Module<Tensor, Tensor> act;
    private readonly Module<Tensor, Tensor>? post_act = null;
    
    public TimestepEmbedding(
        int in_channels,
        int time_embed_dim,
        string act_fn = "silu",
        int? out_dim = null,
        string? post_act_fn = null,
        int? cond_proj_dim = null,
        bool sample_proj_bias = true)
        : base(nameof(TimestepEmbedding))
    {
        this.linear_1 = Linear(in_channels, time_embed_dim, sample_proj_bias);

        if (cond_proj_dim is int proj_dim)
        {
            this.cond_proj = Linear(proj_dim, time_embed_dim, false);
        }

        this.act = Utils.GetActivation(act_fn);

        var time_embed_dim_out = out_dim ?? time_embed_dim;

        this.linear_2 = Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias);

        if (post_act_fn is string post_act_fn_str)
        {
            this.post_act = Utils.GetActivation(post_act_fn_str);
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor sample, Tensor? condition = null)
    {
        if (this.cond_proj is not null && condition is not null)
        {
            sample = sample + this.cond_proj.forward(condition);
        }

        sample = this.linear_1.forward(sample);

        if (this.act is not null)
        {
            sample = this.act.forward(sample);
        }

        sample = this.linear_2.forward(sample);

        if (this.post_act is not null)
        {
            sample = this.post_act.forward(sample);
        }

        return sample;
    }
}