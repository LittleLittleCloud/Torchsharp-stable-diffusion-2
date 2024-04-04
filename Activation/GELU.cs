namespace SD;
public class GELU : Module<Tensor, Tensor>
{
    private readonly Linear proj;
    private readonly string approximate;

    public GELU(
        int dim_in,
        int dim_out,
        string approximate = "none",
        bool bias = true,
        ScalarType dtype = ScalarType.Float32)
        : base("GELU")
    {
        this.proj = Linear(dim_in, dim_out, bias, dtype: dtype);
        this.approximate = approximate;
    }

    public Tensor gelu(Tensor gate)
    {
        // todo
        // support approximate
        return functional.gelu(gate);
    }

    public override Tensor forward(Tensor hidden_states)
    {
        hidden_states = this.proj.forward(hidden_states);
        hidden_states = this.gelu(hidden_states);
        return hidden_states;
    }
}