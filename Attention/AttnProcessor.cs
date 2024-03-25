using System.Reflection;
using SD;
using static TorchSharp.torch;

public abstract class AttnProcessorBase
{
    abstract public Tensor Process(
        CrossAttention attn,
        Tensor hidden_states,
        Tensor? encoder_hidden_states = null,
        Tensor? attention_mask = null,
        Tensor? temb = null);
}

public class AttnProcessor2_0 : AttnProcessorBase
{
    public override Tensor Process(
        CrossAttention attn,
        Tensor hidden_states,
        Tensor? encoder_hidden_states = null,
        Tensor? attention_mask = null,
        Tensor? temb = null)
    {
        var q = attn.ToQ(hidden_states);
        var k = attn.ToK(encoder_hidden_states ?? hidden_states);
        var v = attn.ToV(encoder_hidden_states ?? hidden_states);

        var attn_output = attn.ProcessAttention(q, k, v, attention_mask, temb);

        return attn.ToOut(attn_output);
    }
}