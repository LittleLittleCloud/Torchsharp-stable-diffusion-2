namespace SD;
public class FeedForward : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> dropout;
    private readonly Linear linear_cls;
    private readonly ModuleList<Module<Tensor, Tensor>> net;

    public FeedForward(
        int dim,
        int? dim_out = null,
        int mult = 4,
        double dropout = 0.0,
        string activation_fn = "geglu",
        bool final_dropout = false,
        int? inner_dim = null,
        bool bias = true,
        ScalarType dtype = ScalarType.Float32)
        : base(nameof(FeedForward))
    {
        inner_dim = inner_dim ?? (int)(dim * mult);
        dim_out = dim_out ?? dim;
        var act_fn = new GEGLU(dim, inner_dim.Value, bias, dtype: dtype);
        if (activation_fn == "geglu")
        {
            act_fn = new GEGLU(dim, inner_dim.Value, bias, dtype: dtype);
        }
        else
        {
            throw new NotImplementedException("Only GEGLU is supported for now");
        }

        net = new ModuleList<Module<Tensor, Tensor>>();
        // project in
        net.Add(act_fn);
        // project dropout
        net.Add(nn.Dropout(dropout));
        // project out
        net.Add(Linear(inner_dim.Value, dim_out.Value, bias, dtype: dtype));
        // FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if (final_dropout)
        {
            net.Add(nn.Dropout(dropout));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor hidden_states)
    {
        foreach (var module in net)
        {
            hidden_states = module.forward(hidden_states);
        }
        return hidden_states;
    }
}