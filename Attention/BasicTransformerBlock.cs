using SD;

public class BasicTransformerBlock : Module<Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor>
{
    private readonly int dim;
    private readonly int num_attention_heads;
    private readonly int attention_head_dim;
    private readonly double dropout;
    private readonly int? cross_attention_dim;
    private readonly string activation_fn;
    private readonly int? num_embeds_ada_norm;
    private readonly bool attention_bias;
    private readonly bool only_cross_attention;
    private readonly bool double_self_attention;
    private readonly bool upcast_attention;
    private readonly bool norm_elementwise_affine;
    private readonly string norm_type;
    private readonly double norm_eps;
    private readonly bool final_dropout;
    private readonly string attention_type;
    private readonly string? positional_embeddings;
    private readonly int? num_positional_embeddings;
    private readonly int? ada_norm_continous_conditioning_embedding_dim;
    private readonly int? ada_norm_bias;
    private readonly int? ff_inner_dim;
    private readonly bool ff_bias;
    private readonly bool attention_out_bias;
    private readonly bool use_ada_layer_norm_zero;
    private readonly bool use_ada_layer_norm;
    private readonly bool use_ada_layer_norm_single;
    private readonly bool use_layer_norm;
    private readonly bool use_ada_layer_norm_conitnuous;

    private readonly int? _chunk_size;
    private readonly int _chunk_dim;

    private readonly Module<Tensor, Tensor> norm1;
    private readonly Attention attn1;

    private readonly Module<Tensor, Tensor>? norm2 = null;
    private readonly Attention? attn2 = null;

    private readonly Module<Tensor, Tensor>? norm3 = null;

    private readonly FeedForward ff;

    public BasicTransformerBlock(
        int dim,
        int num_attention_heads,
        int attention_head_dim,
        double dropout = 0.0,
        int? cross_attention_dim = null,
        string activation_fn = "geglu",
        int? num_embeds_ada_norm = null,
        bool attention_bias = false,
        bool only_cross_attention = false,
        bool double_self_attention = false,
        bool upcast_attention = false,
        bool norm_elementwise_affine = true,
        string norm_type = "layer_norm",
        double norm_eps = 1e-5,
        bool final_dropout = false,
        string attention_type = "default",
        string? positional_embeddings = null,
        int? num_positional_embeddings = null,
        int? ada_norm_continous_conditioning_embedding_dim = null,
        int? ada_norm_bias = null,
        int? ff_inner_dim = null,
        bool ff_bias = true,
        bool attention_out_bias = true,
        ScalarType dtype = ScalarType.Float32
    ) : base(nameof(BasicTransformerBlock))
    {
        this.dim = dim;
        this.num_attention_heads = num_attention_heads;
        this.attention_head_dim = attention_head_dim;
        this.dropout = dropout;
        this.cross_attention_dim = cross_attention_dim;
        this.activation_fn = activation_fn;
        this.num_embeds_ada_norm = num_embeds_ada_norm;
        this.attention_bias = attention_bias;
        this.only_cross_attention = only_cross_attention;
        this.double_self_attention = double_self_attention;
        this.upcast_attention = upcast_attention;
        this.norm_elementwise_affine = norm_elementwise_affine;
        this.norm_type = norm_type;
        this.norm_eps = norm_eps;
        this.final_dropout = final_dropout;
        this.attention_type = attention_type;
        this.positional_embeddings = positional_embeddings;
        this.num_positional_embeddings = num_positional_embeddings;
        this.ada_norm_continous_conditioning_embedding_dim = ada_norm_continous_conditioning_embedding_dim;
        this.ada_norm_bias = ada_norm_bias;
        this.ff_inner_dim = ff_inner_dim;
        this.ff_bias = ff_bias;
        this.attention_out_bias = attention_out_bias;

        if (norm_type != "layer_norm")
        {
            throw new NotImplementedException("Only layer_norm is supported for now");
        }

        this.use_ada_layer_norm_zero = false;
        this.use_ada_layer_norm = false;
        this.use_ada_layer_norm_single = false;
        this.use_layer_norm = true;
        this.use_ada_layer_norm_conitnuous = false;

        if (this.positional_embeddings is not null)
        {
            throw new NotImplementedException("Positional embeddings are not supported for now");
        }

        this.norm1 = LayerNorm(dim, elementwise_affine: norm_elementwise_affine, eps: norm_eps, dtype: dtype);
        this.attn1 = new Attention(
            query_dim: dim,
            heads: num_attention_heads,
            dim_head: attention_head_dim,
            dropout: (float)dropout,
            bias: attention_bias,
            cross_attention_dim: only_cross_attention ? cross_attention_dim : null,
            upcast_attention: upcast_attention,
            out_bias: attention_out_bias,
            dtype: dtype);

        if (cross_attention_dim is not null || double_self_attention)
        {
            this.norm2 = LayerNorm(dim, elementwise_affine: norm_elementwise_affine, eps: norm_eps, dtype: dtype);
            this.attn2 = new Attention(
                query_dim: dim,
                cross_attention_dim: double_self_attention ? null : cross_attention_dim,
                heads: num_attention_heads,
                dim_head: attention_head_dim,
                dropout: (float)dropout,
                bias: attention_bias,
                upcast_attention: upcast_attention,
                out_bias: attention_out_bias,
                dtype: dtype);
        }

        if (norm_type == "layer_norm")
        {
            this.norm3 = LayerNorm(dim, elementwise_affine: norm_elementwise_affine, eps: norm_eps, dtype: dtype);
        }

        if (attention_type != "default")
        {
            throw new NotImplementedException("Only default attention is supported for now");
        }

        this.ff = new FeedForward(
            dim: dim,
            dropout: dropout,
            activation_fn: activation_fn,
            final_dropout: final_dropout,
            inner_dim: ff_inner_dim,
            bias: ff_bias,
            dtype: dtype);

        this._chunk_size = null;
        this._chunk_dim = 0;
    }

    public override Tensor forward(
        Tensor hidden_states,
        Tensor? attention_mask = null,
        Tensor? encoder_hidden_states = null,
        Tensor? encoder_attention_mask = null,
        Tensor? timestep = null)
    {
        // self-attention
        var batch_size = hidden_states.shape[0];

        var norm_hidden_states = this.norm1.forward(hidden_states);
        var attn_output = this.attn1.forward(
            norm_hidden_states,
            encoder_hidden_states: this.only_cross_attention ? encoder_hidden_states : null,
            attention_mask: attention_mask);
        
        hidden_states = hidden_states + attn_output;

        if (hidden_states.ndim == 4)
        {
            hidden_states = hidden_states.squeeze(1);
        }

        // cross-attention
        if (this.attn2 is not null)
        {
            norm_hidden_states = this.norm2!.forward(hidden_states);
            attn_output = this.attn2.forward(
                norm_hidden_states,
                encoder_hidden_states: encoder_hidden_states,
                attention_mask: encoder_attention_mask);
            
            hidden_states = hidden_states + attn_output;
        }

        // feed-forward
        norm_hidden_states = this.norm3!.forward(hidden_states);
        var ff_output = this.ff.forward(norm_hidden_states);

        hidden_states = hidden_states + ff_output;
        if(hidden_states.ndim == 4)
        {
            hidden_states = hidden_states.squeeze(1);
        }

        return hidden_states;
    }
}