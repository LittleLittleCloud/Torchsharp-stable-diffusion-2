using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace SD;

public class AdaGroupNorm : Module<Tensor, Tensor, Tensor>
{
    private readonly int embedding_dim;
    private readonly int out_dim;
    private readonly int num_groups;
    private readonly string? act_fn;
    private readonly float eps = 1e-5f;
    private readonly Module<Tensor, Tensor>? act;
    private readonly Linear linear;
    private ScalarType defaultDtype;
    public AdaGroupNorm(
        int embedding_dim,
        int out_dim,
        int num_groups,
        string? act_fn = null,
        float eps = 1e-5f,
        ScalarType dtype = ScalarType.Float32)
        : base(nameof(AdaGroupNorm))
    {
        this.embedding_dim = embedding_dim;
        this.out_dim = out_dim;
        this.num_groups = num_groups;
        this.act_fn = act_fn;
        this.eps = eps;
        this.defaultDtype = dtype;

        this.act = act_fn != null ? Utils.GetActivation(act_fn) : null;
        this.linear = Linear(embedding_dim, out_dim * 2, dtype: dtype);
    }

    public override Tensor forward(Tensor x, Tensor emb)
    {
        if (this.act != null)
        {
            emb = this.act.forward(emb);
        }

        emb = this.linear.forward(emb);
        // emb = emb[:, :, None, None]
        emb = emb.unsqueeze(2).unsqueeze(3);
        // scale, shift = emb.chunk(2, dim=1)
        var chunks = emb.chunk(2, 1);
        var scale = chunks[0];
        var shift = chunks[1];

        x = nn.functional.group_norm(x, this.num_groups, eps: this.eps);
        x = x * (1+scale) + shift;

        return x;
    }
}