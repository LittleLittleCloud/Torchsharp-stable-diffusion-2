using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class CLIPAttention : Module<Tensor, Tensor?, Tensor?, bool?, (Tensor, Tensor?)>
{
    private readonly CLIPTextConfig config;
    private readonly int embed_dim;
    private readonly int num_heads;
    private readonly int head_dim;
    private readonly float scale;
    private readonly double dropout;
    private readonly Linear k_proj;
    private readonly Linear v_proj;
    private readonly Linear q_proj;
    private readonly Linear out_proj;
    private readonly ScalarType dtype;

    public CLIPAttention(
        CLIPTextConfig config)
        : base(nameof(CLIPAttention))
    {
        this.config = config;
        this.embed_dim = config.HiddenSize;
        this.num_heads = config.NumAttentionHeads;
        this.head_dim = this.embed_dim / this.num_heads;
        this.dtype = config.DType;
        if (this.head_dim * this.num_heads != this.embed_dim)
        {
            throw new ArgumentException("embed_dim must be divisible by num_heads");
        }

        this.scale = 1.0f / MathF.Sqrt(this.head_dim);
        this.dropout = config.AttentionDropout;

        this.k_proj = Linear(this.embed_dim, this.embed_dim, dtype: dtype);
        this.v_proj = Linear(this.embed_dim, this.embed_dim, dtype: dtype);
        this.q_proj = Linear(this.embed_dim, this.embed_dim, dtype: dtype);
        this.out_proj = Linear(this.embed_dim, this.embed_dim, dtype: dtype);
        
        RegisterComponents();
    }

    public override (Tensor, Tensor?) forward(
        Tensor hidden_states,
        Tensor? attention_mask = null,
        Tensor? causal_attention_mask = null,
        bool? output_attentions = false)
    {
        // shape of hidden_states: (bsz, time, channel)
        var bsz = (int)hidden_states.shape[0];
        var tgt_len = (int)hidden_states.shape[1];
        var embed_dim = (int)hidden_states.shape[2];

        // get query proj
        var query_states = this.q_proj.forward(hidden_states) * this.scale;
        var key_states = this._shape(this.k_proj.forward(hidden_states), -1, bsz);
        var value_states = this._shape(this.v_proj.forward(hidden_states), -1, bsz);

        long[] proj_shape = [bsz * this.num_heads, -1, this.head_dim];
        query_states = this._shape(query_states, tgt_len, bsz).view(proj_shape);
        key_states = key_states.view(proj_shape);
        value_states = value_states.view(proj_shape);

        var src_len = key_states.shape[1];
        var attn_weights = torch.bmm(query_states, key_states.transpose(1, 2));
        // attn_weights's shape: (bsz * num_heads, tgt_len, src_len)

        if (causal_attention_mask is not null)
        {
            // causal_attention_mask's shape: (bsz, 1, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz, this.num_heads, tgt_len, src_len) + causal_attention_mask;
            attn_weights = attn_weights.view(bsz * this.num_heads, tgt_len, src_len);
        }

        if (attention_mask is not null)
        {
            // attention_mask's shape: (bsz, 1, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz, this.num_heads, tgt_len, src_len) + attention_mask;
            attn_weights = attn_weights.view(bsz * this.num_heads, tgt_len, src_len);
        }

        attn_weights = attn_weights.softmax(-1, dtype: this.config.DType);
        Tensor? attn_weights_reshaped = null;

        if (output_attentions == true)
        {
            // this operation is a bit akward, but it's required to
            // make sure that attn_weights keeps its gradient.
            // In order to do so, attn_weights have to reshaped
            // twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, this.num_heads, tgt_len, src_len);
            attn_weights = attn_weights_reshaped.view(bsz * this.num_heads, tgt_len, src_len);
        }

        var attn_probs = nn.functional.dropout(attn_weights, this.dropout, this.training);
        var attn_output = torch.bmm(attn_probs, value_states);

        // attn_output's shape: (bsz * num_heads, tgt_len, head_dim)
        attn_output = attn_output.view(bsz, this.num_heads, tgt_len, this.head_dim);
        attn_output = attn_output.transpose(1, 2);
        attn_output.Peek("attn_output");
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim);
        attn_output = this.out_proj.forward(attn_output);

        return (attn_output, attn_weights_reshaped);
    }

    private Tensor _shape(Tensor tensor, int seq_len, int bsz)
    {
        return tensor.view(bsz, seq_len, this.num_heads, this.head_dim).permute(0, 2, 1, 3).contiguous();
    }
}