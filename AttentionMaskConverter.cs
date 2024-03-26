using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class AttentionMaskConverter
{
    private readonly bool is_casual;
    private readonly int? sliding_window;

    public AttentionMaskConverter(bool is_casual, int? sliding_window)
    {
        this.is_casual = is_casual;
        this.sliding_window = sliding_window;
    }

    public Tensor? ToCasual4D(
        int batch_size,
        int query_length,
        int key_value_length,
        ScalarType dtype,
        Device device)
    {
        if (!is_casual)
        {
            throw new ArgumentException("This is not a casual mask");
        }

        long[] input_shape = [batch_size, query_length];
        var past_key_values_length = key_value_length - query_length;

        // create causal mask
        // [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        Tensor? casual_4d_mask = null;
        if (query_length > 1 || this.sliding_window is int window)
        {
            casual_4d_mask = MakeCasualMask(input_shape, dtype, device, past_key_values_length, this.sliding_window);
        }

        return casual_4d_mask;
    }

    public static Tensor MakeCasualMask(
        long[] input_ids_shape,
        ScalarType dtype,
        Device device,
        int past_key_values_length = 0,
        int? sliding_window = null)
    {
        // Make causal mask used for bi-directional self-attention.
        var bsz = input_ids_shape[0];
        var tgt_len = input_ids_shape[1];
        var mask = torch.full([tgt_len, tgt_len], torch.finfo(dtype).min, dtype: dtype, device: device);
        var mask_cond = torch.arange(tgt_len, device: device);
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(tgt_len, 1), 0);
        mask = mask.to(dtype);

        if (past_key_values_length > 0)
        {
            mask = torch.cat([torch.zeros([tgt_len, past_key_values_length], dtype: dtype, device: device), mask], dim: -1);
        }

        if (sliding_window is int window)
        {
            var diagonal = past_key_values_length - window - 1;
            var context_mask = torch.tril(torch.ones([tgt_len, tgt_len], dtype: ScalarType.Bool, device: device), diagonal: diagonal);
            mask = mask.masked_fill(context_mask, torch.finfo(dtype).min);
        }

        // return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

        return mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, tgt_len, tgt_len + past_key_values_length);
    }

    /// <summary>
    /// Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`
    /// </summary>
    /// <param name="input_shape">The input shape should be a tuple that defines `(batch_size, query_length)`.</param>
    public static Tensor? Create4DCasualAttentionMask(
        long[] input_shape,
        ScalarType dtype,
        Device device,
        int past_key_values_length = 0,
        int? sliding_window = null)
    {
        var batch_size = (int)input_shape[0];
        var query_length = (int)input_shape[1];
        var converter = new AttentionMaskConverter(is_casual: true, sliding_window: sliding_window);
        var key_value_length = past_key_values_length + query_length;
        return converter.ToCasual4D(batch_size, query_length, key_value_length, dtype, device);
    }

    public static Tensor ExpandMask(
        Tensor mask,
        ScalarType dtype,
        int? tgt_len = null)
    {
        var bsz = (int)mask.shape[0];
        var src_len = (int)mask.shape[1];
        tgt_len = tgt_len ?? src_len;

        var expanded_mask = mask.unsqueeze(1).unsqueeze(1).expand(bsz, 1, tgt_len.Value, src_len).to(dtype);
        var inverted_mask = 1.0 - expanded_mask;

        return inverted_mask.masked_fill(inverted_mask.to(ScalarType.Bool), torch.finfo(dtype).min);
    }
}