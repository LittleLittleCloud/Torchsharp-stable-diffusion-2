using FluentAssertions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using SD;

public static class Utils
{
    public static Tensor ApplyRotaryEmbeddings(Tensor input, Tensor freqsComplex)
    {
        // Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
        // Two consecutive values will become a single complex number
        // (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
        var input_complex = input.to_type(ScalarType.Float32).reshape(input.shape[0], input.shape[1], input.shape[2], -1, 2).view_as_complex();
        freqsComplex = freqsComplex.to(input.device);

        // Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
        // (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
        var freqs_complex_reshaped = freqsComplex.unsqueeze(0).unsqueeze(2);

        // Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
        // Which results in the rotation of the complex number as shown in the Figure 1 of the paper
        // (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
        var rotated_complex = input_complex * freqs_complex_reshaped;
        // Console.WriteLine(rotated_complex.mean().ToSingle());

        // Convert the complex number back to the real number
        // (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
        var rotated = rotated_complex.view_as_real();

        // (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
        var rotated_reshaped = rotated.reshape(rotated.shape[0], rotated.shape[1], rotated.shape[2], -1);

        input.shape.Should().BeEquivalentTo(rotated_reshaped.shape);
        return rotated_reshaped.type_as(input);
    }

    public static Module<Tensor, Tensor> GetActivation(string act_fn)
    {
        return act_fn switch
        {
            "silu" => nn.SiLU(),
            "relu" => nn.ReLU(),
            "gelu" => nn.GELU(),
            "tanh" => nn.Tanh(),
            "swish" => nn.SiLU(),
            _ => throw new ArgumentException("Invalid activation function", nameof(act_fn)),
        };
    }

    /// <summary>
    /// This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    /// </summary>
    /// <param name="timesteps">a 1-D Tensor of N indices, one per batch element. These may be fractional</param>
    /// <param name="embedding_dim">the dimension of the output.</param>
    /// <param name="flip_sin_to_cos"></param>
    /// <param name="downscale_freq_shift"></param>
    /// <param name="scale"></param>
    /// <param name="max_period">controls the minimum frequency of the embeddings.</param>
    /// <returns>an [N x dim] Tensor of positional embeddings.</returns>
    public static Tensor GetTimestepEmbedding(
        Tensor timesteps,
        int embedding_dim,
        bool flip_sin_to_cos = false,
        float downscale_freq_shift = 1,
        float scale = 1f,
        int max_period = 10000)
    {
        var half_dim = embedding_dim / 2;
        var exponent = -Math.Log(max_period) * torch.arange(0, half_dim, device: timesteps.device, dtype: ScalarType.Float32);
        exponent = exponent / (half_dim - downscale_freq_shift);

        var emb = torch.exp(exponent);
        emb = timesteps.unsqueeze(1).to(ScalarType.Float32) * emb.unsqueeze(0);

        emb = scale * emb;
        emb = torch.cat([emb.sin(), emb.cos()], dim: 1);

        if (flip_sin_to_cos)
        {
            emb = torch.cat([emb[.., half_dim..], emb[.., ..half_dim]], dim: -1);
        }

        if (embedding_dim % 2 == 1)
        {
            emb = nn.functional.pad(emb, [0, 1, 0, 0]);
        }

        return emb;
    }

    public static Module<DownBlock2DInput, DownBlock2DOutput> GetDownBlock(
        string down_block_type,
        int num_layers,
        int in_channels,
        int out_channels,
        int temb_channels,
        bool add_downsample,
        float resnet_eps,
        string resnet_act_fn,
        int transformer_layers_per_block = 1,
        int? num_attention_heads = null,
        int? resnet_groups = null,
        int? cross_attention_dim = null,
        int? downsample_padding = null,
        bool dual_cross_attention = false,
        bool use_linear_projection = false,
        bool only_cross_attention = false,
        bool upcast_attention = false,
        string resnet_time_scale_shift = "default",
        string attention_type = "default",
        bool resnet_skip_time_act = false,
        float resnet_out_scale_factor = 1.0f,
        string? cross_attention_norm = null,
        int? attention_head_dim = null,
        string? downsample_type = null,
        float dropout = 0.0f)
    {
        // If attn head dim is not defined, we default it to the number of heads
        attention_head_dim ??= num_attention_heads;

        down_block_type = down_block_type.StartsWith("UNetRes") ? down_block_type.Substring(7) : down_block_type;

        if (down_block_type == nameof(DownBlock2D))
        {
            return new DownBlock2D(
                num_layers: num_layers,
                in_channels: in_channels,
                out_channels: out_channels,
                temb_channels: temb_channels,
                dropout: dropout,
                add_downsample: add_downsample,
                resnet_eps: resnet_eps,
                resnet_act_fn: resnet_act_fn,
                resnet_groups: resnet_groups,
                downsample_padding: downsample_padding,
                resnet_time_scale_shift: resnet_time_scale_shift);
        }
        else if (down_block_type == nameof(CrossAttnDownBlock2D))
        {
            if (cross_attention_dim is null)
            {
                throw new ArgumentException("Cross attention dimension must be defined for CrossAttnDownBlock2D", nameof(cross_attention_dim));
            }

            return new CrossAttnDownBlock2D(
                num_layers: num_layers,
                // transformer_layers_per_block: transformer_layers_per_block,
                in_channels: in_channels,
                out_channels: out_channels,
                temb_channels: temb_channels,
                dropout: dropout,
                add_downsample: add_downsample,
                resnet_eps: resnet_eps,
                resnet_act_fn: resnet_act_fn,
                resnet_groups: resnet_groups,
                downsample_padding: downsample_padding,
                cross_attention_dim: cross_attention_dim,
                num_attention_heads: num_attention_heads,
                dual_cross_attention: dual_cross_attention,
                use_linear_projection: use_linear_projection,
                only_cross_attention: only_cross_attention,
                upcast_attention: upcast_attention,
                resnet_time_scale_shift: resnet_time_scale_shift,
                attention_type: attention_type);
        }
        else
        {
            throw new ArgumentException("Invalid down block type", nameof(down_block_type));
        };
    }
    public static Tensor PrecomputeThetaPosFrequencies(int headDim, int seqLen, string device, float theta = 10000.0f)
    {
        // As written in the paragraph 3.2.2 of the paper
        // >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
        if (headDim % 2 != 0)
        {
            throw new ArgumentException("Dimension must be divisible by 2", nameof(headDim));
        }

        // Build the theta parameter
        // According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
        // Shape: (Head_Dim / 2)
        var thetaNumerator = torch.arange(0, headDim, 2).to(torch.float32).to(device);
        // Shape: (Head_Dim / 2)
        var thetaInput = torch.pow(theta, -1.0f * (thetaNumerator / headDim)).to(device); // (Dim / 2)
        // Construct the positions (the "m" parameter)
        // Shape: (Seq_Len)
        var m = torch.arange(seqLen, device: device);
        // Multiply each theta by each position using the outer product.
        // Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        var freqs = torch.outer(m, thetaInput).to(torch.float32).to(device);

        // We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
        // (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        var freqsComplex = torch.polar(torch.ones_like(freqs), freqs);

        return freqsComplex;
    }

    public static Tensor RotateHalf(Tensor x)
    {
        var x1 = x[.., .., .., ..(int)(x.shape[^1] / 2)];
        var x2 = x[.., .., .., (int)(x.shape[^1] / 2)..];
        // (x1 * x1 * x2).Peek("x1 * x1 * x2");
        return torch.cat([-x2, x1], dim: -1);
    }

    public static (Tensor, Tensor) ApplyRotaryPosEmb(Tensor q, Tensor k, Tensor cos, Tensor sin, Tensor positionIds, int unsqueezeDim = 1)
    {
        // The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
        // sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
        // that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
        // k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
        // cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
        // the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

        cos = cos[positionIds].unsqueeze(unsqueezeDim);
        sin = sin[positionIds].unsqueeze(unsqueezeDim);
        var qEmbed = q * cos;
        qEmbed += RotateHalf(q) * sin;

        var kEmbed = k * cos;
        kEmbed += RotateHalf(k) * sin;
        // var kEmbed = (k * cos) + (RotateHalf(k) * sin);
        return (qEmbed, kEmbed);
    }


    public static Tensor RepeatKV(Tensor x, int nRep)
    {
        var batchSize = x.shape[0];
        var seqLen = x.shape[1];
        var nKVHeads = x.shape[2];
        var headDim = x.shape[3];
        if (nRep == 1)
        {
            return x;
        }

        return x.unsqueeze(3)
                .expand(batchSize, seqLen, nKVHeads, nRep, headDim)
                .view(batchSize, seqLen, nKVHeads * nRep, headDim);
    }

}
