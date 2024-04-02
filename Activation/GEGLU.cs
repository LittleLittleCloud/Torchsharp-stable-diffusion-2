// class GEGLU(nn.Module):
//     r"""
//     A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

//     Parameters:
//         dim_in (`int`): The number of channels in the input.
//         dim_out (`int`): The number of channels in the output.
//         bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
//     """

//     def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
//         super().__init__()
//         self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

//     def gelu(self, gate: torch.Tensor) -> torch.Tensor:
//         if gate.device.type != "mps":
//             return F.gelu(gate)
//         # mps: gelu is not implemented for float16
//         return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

//     def forward(self, hidden_states, *args, **kwargs):
//         if len(args) > 0 or kwargs.get("scale", None) is not None:
//             deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
//             deprecate("scale", "1.0.0", deprecation_message)

//         hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
//         return hidden_states * self.gelu(gate)

namespace SD;

public class GEGLU : Module<Tensor, Tensor>
{
    private readonly Linear proj;

    public GEGLU(int dim_in, int dim_out, bool bias = true)
        : base("GEGLU")
    {
        this.proj = Linear(dim_in, dim_out * 2, bias);
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