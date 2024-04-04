using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class CLIPEncoderLayer : Module<Tensor, Tensor?, Tensor?, bool?, (Tensor, Tensor?)>
{
    private readonly int embed_dim;
    private readonly CLIPAttention self_attn;
    private readonly LayerNorm layer_norm1;
    private readonly CLIPMLP mlp;
    private readonly LayerNorm layer_norm2;

    public CLIPEncoderLayer(CLIPTextConfig config)
        : base(nameof(CLIPEncoderLayer))
    {
        this.embed_dim = config.HiddenSize;
        this.self_attn = new CLIPAttention(config);
        this.layer_norm1 = LayerNorm(embed_dim, eps: config.LayerNormEps, dtype: config.DType);
        this.mlp = new CLIPMLP(config);
        this.layer_norm2 = LayerNorm(embed_dim, eps: config.LayerNormEps, dtype: config.DType);

        RegisterComponents();
    }

    public override (Tensor, Tensor?) forward(
        Tensor hidden_states,
        Tensor? attention_mask = null,
        Tensor? causal_attention_mask = null,
        bool? output_attentions = false)
    {
        var residual = hidden_states;
        hidden_states = this.layer_norm1.forward(hidden_states);
        (hidden_states, var attention_weights) = this.self_attn.forward(hidden_states, attention_mask, causal_attention_mask, output_attentions);
        hidden_states = hidden_states + residual;
        residual = hidden_states;
        hidden_states = this.layer_norm2.forward(hidden_states);
        hidden_states = this.mlp.forward(hidden_states);
        hidden_states = hidden_states + residual;
        if (output_attentions == true)
        {
            return (hidden_states, attention_weights);
        }
        else
        {
            return (hidden_states, null);
        }
    }
}