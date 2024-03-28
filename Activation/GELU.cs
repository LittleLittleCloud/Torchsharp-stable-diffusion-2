// class GELU(nn.Module):
//     r"""
//     GELU activation function with tanh approximation support with `approximate="tanh"`.

//     Parameters:
//         dim_in (`int`): The number of channels in the input.
//         dim_out (`int`): The number of channels in the output.
//         approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
//         bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
//     """

//     def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
//         super().__init__()
//         self.proj = nn.Linear(dim_in, dim_out, bias=bias)
//         self.approximate = approximate

//     def gelu(self, gate: torch.Tensor) -> torch.Tensor:
//         if gate.device.type != "mps":
//             return F.gelu(gate, approximate=self.approximate)
//         # mps: gelu is not implemented for float16
//         return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

//     def forward(self, hidden_states):
//         hidden_states = self.proj(hidden_states)
//         hidden_states = self.gelu(hidden_states)
//         return hidden_states
namespace SD;
public class GELU : Module<Tensor, Tensor>
{
    private readonly Linear proj;
    private readonly string approximate;

    public GELU(
        int dim_in,
        int dim_out,
        string approximate = "none",
        bool bias = true)
        : base("GELU")
    {
        this.proj = Linear(dim_in, dim_out, bias);
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