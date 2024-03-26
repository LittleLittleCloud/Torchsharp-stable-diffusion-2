using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace SD;

public class CLIPMLP : Module<Tensor, Tensor>
{
    private readonly CLIPTextConfig config;

    private readonly Linear fc1;
    private readonly Linear fc2;
    private readonly Module<Tensor, Tensor> activation_fn;

    public CLIPMLP(CLIPTextConfig config)
        : base(nameof(CLIPMLP))
    {
        this.config = config;
        this.activation_fn = Utils.GetActivation(config.HiddenAct);
        this.fc1 = Linear(config.HiddenSize, config.IntermediateSize);
        this.fc2 = Linear(config.IntermediateSize, config.HiddenSize);
        RegisterComponents();
    }

    public override Tensor forward(Tensor hidden_states)
    {
        hidden_states = this.fc1.forward(hidden_states);
        hidden_states = this.activation_fn.forward(hidden_states);
        hidden_states = this.fc2.forward(hidden_states);

        return hidden_states;
    }
}