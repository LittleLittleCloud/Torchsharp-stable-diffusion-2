namespace SD;

public class GEGLU : Module<Tensor, Tensor>
{
    private readonly Linear proj;

    public GEGLU(int dim_in, int dim_out, bool bias = true, ScalarType dtype = ScalarType.Float32)
        : base("GEGLU")
    {
        this.proj = Linear(dim_in, dim_out * 2, bias, dtype: dtype);
    }

    public Tensor gelu(Tensor gate)
    {
        return functional.gelu(gate);
    }

    public override Tensor forward(Tensor hidden_states)
    {
        var chunks = this.proj.forward(hidden_states).chunk(2, -1);
        hidden_states = chunks[0];
        var gate = chunks[1];
        return hidden_states * this.gelu(gate);
    }
}