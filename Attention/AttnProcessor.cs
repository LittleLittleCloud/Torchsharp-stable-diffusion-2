using System.Reflection;
using SD;
using static TorchSharp.torch;

public abstract class AttnProcessorBase
{
    abstract public Tensor Process(
        Attention attn,
        Tensor hidden_states,
        Tensor? encoder_hidden_states = null,
        Tensor? attention_mask = null,
        Tensor? temb = null);
}

public class AttnProcessor2_0 : AttnProcessorBase
{
    public override Tensor Process(
        Attention attn,
        Tensor hidden_states,
        Tensor? encoder_hidden_states = null,
        Tensor? attention_mask = null,
        Tensor? temb = null)
    {
        var residual = hidden_states;
        if (attn.SpatialNorm is not null){
            hidden_states = attn.SpatialNorm.forward(hidden_states, temb);
        }

        var input_ndim = hidden_states.ndim;
        int batch_size;
        long channel = 0;
        long height = 0;
        long width = 0;
        if (input_ndim == 4){
            batch_size = (int)hidden_states.shape[0];
            channel = hidden_states.shape[1];
            height = hidden_states.shape[2];
            width = hidden_states.shape[3];
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2);
        }

        int sequence_length;
        if (encoder_hidden_states is not null){
            batch_size = (int)encoder_hidden_states.shape[0];
            sequence_length = (int)encoder_hidden_states.shape[1];
        }
        else{
            batch_size = (int)hidden_states.shape[0];
            sequence_length = (int)hidden_states.shape[1];
        }
        
        if (attention_mask is not null)
        {
            attention_mask = attn.PrepareAttentionMask(attention_mask, sequence_length, batch_size);
            attention_mask = attention_mask!.view(batch_size, attn.Heads, -1, attention_mask.shape[^1]);
        }

        if (attn.GroupNorm is not null)
        {
            hidden_states = attn.GroupNorm.forward(hidden_states.transpose(1, 2)).transpose(1, 2);
        }

        var query = attn.ToQ.forward(hidden_states);

        if (encoder_hidden_states is null)
        {
            encoder_hidden_states = hidden_states;
        }
        else if (attn.NormCross is not null)
        {
            encoder_hidden_states = attn.NormEncoderHiddenStates(encoder_hidden_states);
        }

        var key = attn.ToK.forward(encoder_hidden_states);
        var value = attn.ToV.forward(encoder_hidden_states);
        var inner_dim = key.shape[^1];
        var head_dim = inner_dim / attn.Heads;
        query = query.view(batch_size, -1, attn.Heads, head_dim).transpose(1, 2);
        key = key.view(batch_size, -1, attn.Heads, head_dim).transpose(1, 2);
        value = value.view(batch_size, -1, attn.Heads, head_dim).transpose(1, 2);
        hidden_states = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask: attention_mask, p: 0, is_casual: false);
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.Heads * head_dim);
        hidden_states = hidden_states.to(query.dtype);

        // linear proj
        hidden_states = attn.ToOut[0].forward(hidden_states);
        // dropout
        hidden_states = attn.ToOut[1].forward(hidden_states);

        if (input_ndim == 4)
        {
            hidden_states = hidden_states.transpose(-1, -2).view(batch_size, channel, height, width);
        }

        if (attn.ResidualConnection)
        {
            hidden_states = hidden_states + residual;
        }

        hidden_states = hidden_states / attn.RescaleOutputFactor;

        return hidden_states;
    }
}