using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class Attention : Module<Tensor, Tensor?, Tensor?, Tensor?, Tensor>
{
    private readonly int inner_dim;
    private readonly int query_dim;
    private readonly bool use_bias;
    private readonly bool is_cross_attention;
    private readonly int cross_attention_dim;
    private readonly bool upcast_attention;
    private readonly bool upcast_softmax;
    private readonly float rescale_output_factor;
    private readonly bool residual_connection;
    private readonly float dropout;
    private readonly bool fused_projections;
    private readonly int out_dim;
    private readonly bool scale_qk;
    private readonly float scale;
    private readonly int heads;
    private readonly int sliceable_head_dim;
    private readonly int? added_kv_proj_dim;
    private readonly bool only_cross_attention;

    private readonly AttnProcessorBase processor;

    private Module<Tensor, Tensor>? group_norm;
    private Module<Tensor, Tensor?, Tensor>? spatial_norm;
    private Module<Tensor, Tensor>? norm_cross;
    private Module? add_k_proj;
    private Module? add_v_proj;
    private Module<Tensor, Tensor> to_q;
    private Module<Tensor, Tensor> to_k;
    private Module<Tensor, Tensor> to_v;
    private ModuleList<Module<Tensor, Tensor>> to_out;

    // deprecated module
    private Module<Tensor, Tensor>? query;
    private Module<Tensor, Tensor>? key;
    private Module<Tensor, Tensor>? value;
    private Module<Tensor, Tensor>? proj_attn;

    /// <summary>
    /// Cross-Attention module.
    /// </summary>
    /// <param name="query_dim">The number of channels in the query.</param>
    /// <param name="cross_attention_dim">The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.</param>
    /// <param name="heads">The number of heads to use for multi-head attention.</param>
    /// <param name="dim_head">The number of channels in each head.</param>
    /// <param name="dropout">The dropout probability to use.</param>
    /// <param name="bias">Set to `True` for the query, key, and value linear layers to contain a bias parameter.</param>
    /// <param name="upcast_attention">Set to `True` to upcast the attention computation to `float32`.</param>
    /// <param name="upcast_softmax">Set to `True` to upcast the softmax computation to `float32`.</param>
    /// <param name="cross_attention_norm">The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.</param>
    /// <param name="cross_attention_norm_num_groups">The number of groups to use for the group norm in the cross attention.</param>
    /// <param name="added_kv_proj_dim">The number of channels to use for the added key and value projections. If `None`, no projection is used.</param>
    /// <param name="norm_num_groups">The number of groups to use for the group norm in the attention.</param>
    /// <param name="spatial_norm_dim">The number of channels to use for the spatial normalization.</param>
    /// <param name="out_bias">Set to `True` to use a bias in the output linear layer.</param>
    /// <param name="scale_qk">Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.</param>
    /// <param name="only_cross_attention">Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if `added_kv_proj_dim` is not `None`.</param>
    /// <param name="eps">An additional value added to the denominator in group normalization that is used for numerical stability.</param>
    /// <param name="rescale_output_factor">A factor to rescale the output by dividing it with this value.</param>
    /// <param name="residual_connection">Set to `True` to add the residual connection to the output.</param>
    /// <param name="processor">The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and `AttnProcessor` otherwise.</param>
    /// <param name="out_dim">output dim</param>
    public Attention(
        int query_dim,
        int? cross_attention_dim = null,
        int heads = 8,
        int dim_head = 64,
        float dropout = 0.0f,
        bool bias = false,
        bool upcast_attention = false,
        bool upcast_softmax = false,
        string? cross_attention_norm = null,
        int cross_attention_norm_num_groups = 32,
        int? added_kv_proj_dim = null,
        int? norm_num_groups = null,
        int? spatial_norm_dim = null,
        bool out_bias = true,
        bool scale_qk = true,
        bool only_cross_attention = false,
        float eps = 1e-5f,
        float rescale_output_factor = 1f,
        bool residual_connection = false,
        AttnProcessorBase? processor = null,
        int? out_dim = null,
        bool _from_deprecated_attn_block = false)
        : base(nameof(Attention))
        {
            this.inner_dim = out_dim ?? dim_head * heads;
            this.query_dim = query_dim;
            this.use_bias = bias;
            this.is_cross_attention = cross_attention_dim != null;
            this.cross_attention_dim = cross_attention_dim ?? query_dim;
            this.upcast_attention = upcast_attention;
            this.upcast_softmax = upcast_softmax;
            this.rescale_output_factor = rescale_output_factor;
            this.residual_connection = residual_connection;
            this.dropout = dropout;
            this.fused_projections = false;
            this.out_dim = out_dim ?? query_dim;

            this.scale_qk = scale_qk;
            this.scale = this.scale_qk ? 1.0f / MathF.Sqrt(dim_head) : 1.0f;
            this.heads = out_dim != null ? out_dim.Value / dim_head : heads;
            this.sliceable_head_dim = heads;
            this.added_kv_proj_dim = added_kv_proj_dim;
            this.only_cross_attention = only_cross_attention;

            if (this.added_kv_proj_dim != null && this.only_cross_attention)
            {
                throw new ArgumentException("only_cross_attention can only be set to True if added_kv_proj_dim is not None.");
            }

            if (norm_num_groups != null)
            {
                this.group_norm = GroupNorm(num_channels: query_dim, num_groups: norm_num_groups.Value, eps: eps, affine: true);
            }

            if (spatial_norm_dim != null)
            {
                this.spatial_norm = new SpatialNorm(f_channels: query_dim, zq_channels: spatial_norm_dim.Value);
            }

            if (cross_attention_norm == "layer_norm")
            {
                this.norm_cross = LayerNorm(this.cross_attention_dim);
            }
            else if (cross_attention_norm == "group_norm")
            {
                var norm_cross_num_channels = this.added_kv_proj_dim ?? this.cross_attention_dim;
                this.norm_cross = GroupNorm(num_channels: norm_cross_num_channels, num_groups: cross_attention_norm_num_groups, eps: eps, affine: true);
            }
            else if (cross_attention_norm != null)
            {
                throw new ArgumentException($"cross_attention_norm must be None, layer_norm, or group_norm, got {cross_attention_norm}.");
            }
            
            this.processor = processor ?? new AttnProcessor2_0();

            if (_from_deprecated_attn_block)
            {
                this.query = Linear(query_dim, this.inner_dim, hasBias: this.use_bias);
                if (!this.only_cross_attention)
                {
                    this.key = Linear(this.cross_attention_dim, this.inner_dim, hasBias: this.use_bias);
                    this.value = Linear(this.cross_attention_dim, this.inner_dim, hasBias: this.use_bias);
                }

                if (this.added_kv_proj_dim != null)
                {
                    this.add_k_proj = Linear(this.added_kv_proj_dim.Value, this.inner_dim);
                    this.add_v_proj = Linear(this.added_kv_proj_dim.Value, this.inner_dim);
                }
                this.key = Linear(this.cross_attention_dim, this.inner_dim, hasBias: this.use_bias);
                this.value = Linear(this.cross_attention_dim, this.inner_dim, hasBias: this.use_bias);
                this.proj_attn = Linear(this.inner_dim, this.out_dim, hasBias: out_bias);
                RegisterComponents();

                this.to_q = this.query;
                this.to_k = this.key;
                this.to_v = this.value;
                this.to_out = new ModuleList<Module<Tensor, Tensor>>();
                this.to_out.Add(this.proj_attn);
                this.to_out.Add(nn.Dropout(dropout));
                return;
            }
            else
            {
                this.to_q = Linear(query_dim, this.inner_dim, hasBias: this.use_bias);

                if (!this.only_cross_attention){
                    this.to_k = Linear(this.cross_attention_dim, this.inner_dim, hasBias: this.use_bias);
                    this.to_v = Linear(this.cross_attention_dim, this.inner_dim, hasBias: this.use_bias);
                }

                if (this.added_kv_proj_dim != null)
                {
                    this.add_k_proj = Linear(this.added_kv_proj_dim.Value, this.inner_dim);
                    this.add_v_proj = Linear(this.added_kv_proj_dim.Value, this.inner_dim);
                }

                this.to_out = new ModuleList<Module<Tensor, Tensor>>();
                this.to_out.Add(nn.Linear(this.inner_dim, this.out_dim, hasBias: out_bias));
                this.to_out.Add(nn.Dropout(dropout));
                RegisterComponents();
            }
        }

    public int InnerDim => inner_dim;
    public int QueryDim => query_dim;
    public bool UseBias => use_bias;
    public bool IsCrossAttention => is_cross_attention;
    public int CrossAttentionDim => cross_attention_dim;
    public bool UpcastAttention => upcast_attention;
    public bool UpcastSoftmax => upcast_softmax;
    public float RescaleOutputFactor => rescale_output_factor;
    public bool ResidualConnection => residual_connection;
    public float Dropout => dropout;
    public bool FusedProjections => fused_projections;
    public int OutDim => out_dim;
    public bool ScaleQk => scale_qk;
    public float Scale => scale;
    public int Heads => heads;
    public int SliceableHeadDim => sliceable_head_dim;
    public int? AddedKvProjDim => added_kv_proj_dim;
    public bool OnlyCrossAttention => only_cross_attention;
    public Module<Tensor, Tensor>? GroupNorm => group_norm;
    public Module<Tensor, Tensor?, Tensor>? SpatialNorm => spatial_norm;
    public Module? NormCross => norm_cross;
    public Module? AddKProj => add_k_proj;
    public Module? AddVProj => add_v_proj;
    public Module<Tensor, Tensor> ToQ => to_q;
    public Module<Tensor, Tensor> ToK => to_k;
    public Module<Tensor, Tensor> ToV => to_v;
    public ModuleList<Module<Tensor, Tensor>> ToOut => to_out;
    public AttnProcessorBase? Processor => processor;

    public override Tensor forward(
        Tensor hidden_states,
        Tensor? encoder_hidden_states = null,
        Tensor? attention_mask = null,
        Tensor? temb = null)
    {
        return this.processor.Process(this, hidden_states, encoder_hidden_states, attention_mask, temb);
    }

    /// <summary>
    /// Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads` is the number of heads initialized while constructing the `Attention` class.
    /// </summary>
    public Tensor BatchToHeadDim(Tensor tensor)
    {
        var head_size = this.heads;
        var batch_size = tensor.shape[0];
        var seq_len = tensor.shape[1];
        var dim = tensor.shape[2];
        tensor = tensor.view(new long[] { batch_size / head_size, head_size, seq_len, dim });
        tensor = tensor.permute(0, 2, 1, 3);
        tensor = tensor.reshape(new long[] { batch_size / head_size, seq_len, head_size * dim });
        return tensor;
    }

    /// <summary>
    ///  Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is the number of heads initialized while constructing the `Attention` class.
    /// </summary>
    public Tensor HeadToBatchDim(
        Tensor tensor,
        int out_dim = 3)
    {
        var head_size = this.heads;
        long batch_size;
        long seq_len;
        long dim;
        long extra_dim;
        if (tensor.ndim == 3)
        {
            batch_size = tensor.shape[0];
            seq_len = tensor.shape[1];
            dim = tensor.shape[2];
            extra_dim = 1;
        }
        else if (tensor.ndim == 4)
        {
            batch_size = tensor.shape[0];
            extra_dim = tensor.shape[1];
            seq_len = tensor.shape[2];
            dim = tensor.shape[3];
        }
        else
        {
            throw new ArgumentException("Input tensor must have 3 or 4 dimensions.");
        }

        tensor = tensor.view(new long[] { batch_size, seq_len * extra_dim, head_size, dim / head_size });
        tensor = tensor.permute(0, 2, 1, 3);

        if (out_dim == 3)
        {
            tensor = tensor.reshape(new long[] { batch_size * head_size, seq_len * extra_dim, dim / head_size });
        }

        return tensor;
    }

    /// <summary>
    /// Compute the attention scores.
    /// </summary>
    /// <param name="query"></param>
    /// <param name="key"></param>
    /// <param name="attention_mask"></param>
    /// <returns></returns>
    public Tensor GetAttentionScores(
        Tensor query,
        Tensor key,
        Tensor? attention_mask = null)
    {
        var dtype = query.dtype;
        if (this.upcast_attention)
        {
            query = query.to(ScalarType.Float32);
            key = key.to(ScalarType.Float32);
        }

        Tensor baddbmm_input;
        int beta;
        if (attention_mask is not null)
        {
            baddbmm_input = torch.empty(new long[] { query.shape[0], query.shape[1], key.shape[1] }, dtype: dtype, device: query.device);
            beta = 0;
        }
        else
        {
            baddbmm_input = attention_mask!;
            beta = 1;
        }

        var attention_scores = baddbmm(baddbmm_input, query, key.transpose(-1, -2), beta: beta, alpha: this.scale);
        baddbmm_input.Dispose();

        if (this.upcast_softmax)
        {
            attention_scores = attention_scores.to(ScalarType.Float32);
        }
        var attention_probls = attention_scores.softmax(-1);
        attention_scores.Dispose();

        attention_probls = attention_probls.to(dtype);
        return attention_probls;
    }

    /// <summary>
    /// Prepare the attention mask for the attention computation.
    /// </summary>
    /// <param name="attention_mask">The attention mask to prepare.</param>
    /// <param name="target_length">The target length of the attention mask. This is the length of the attention mask after padding.</param>
    /// <param name="batch_size">The batch size, which is used to repeat the attention mask.</param>
    /// <param name="out_dim">The output dimension of the attention mask. Can be either `3` or `4`.</param>
    /// <returns></returns>
    public Tensor? PrepareAttentionMask(
        Tensor? attention_mask,
        int target_length,
        int batch_size,
        int out_dim = 3)
    {
        var head_size = this.heads;
        if (attention_mask is null)
        {
            return attention_mask;
        }

        var current_length = attention_mask.shape[1];
        if (current_length != target_length)
        {
            attention_mask = nn.functional.pad(attention_mask, [0, target_length], value: 0.0f);
        }

        if (out_dim == 3)
        {
            if (attention_mask.shape[0] < batch_size * head_size)
            {
                attention_mask = attention_mask.repeat_interleave(head_size, 0);
            }
        }
        else if (out_dim == 4)
        {
            attention_mask = attention_mask.unsqueeze(1);
            attention_mask = attention_mask.repeat_interleave(head_size, 1);
        }

        return attention_mask;
    }

    /// <summary>
    /// Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the `Attention` class.
    /// </summary>
    public Tensor NormEncoderHiddenStates(Tensor encoder_hidden_states)
    {
        if (this.norm_cross is null)
        {
            throw new ArgumentException("norm_cross must be specified when constructing the Attention class.");
        }

        if (this.norm_cross is LayerNorm)
        {
            encoder_hidden_states = this.norm_cross.forward(encoder_hidden_states);
        }
        else if (this.norm_cross is GroupNorm)
        {
            // Group norm norms along the channels dimension and expects
            // input to be in the shape of (N, C, *). In this case, we want
            // to norm along the hidden dimension, so we need to move
            // (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2);
            encoder_hidden_states = this.norm_cross.forward(encoder_hidden_states);
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2);
        }
        else
        {
            throw new ArgumentException("norm_cross must be either LayerNorm or GroupNorm.");
        }

        return encoder_hidden_states;
    }
}