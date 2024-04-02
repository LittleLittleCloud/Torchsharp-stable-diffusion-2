public class Transformer2DModel : Module<Tensor, Tensor?, Tensor?,Tensor?, Tensor?, Tensor?, Transformer2DModelOutput>
{
    private readonly int num_attention_heads;
    private readonly int attention_head_dim;
    private readonly int? in_channels;
    private readonly int? out_channels;
    private readonly int num_layers;
    private readonly double dropout;
    private readonly int norm_num_groups;
    private readonly int? cross_attention_dim;
    private readonly bool attention_bias;
    private readonly int? sample_size;
    private readonly int? num_vector_embeds;
    private readonly int? patch_size;
    private readonly string activation_fn;
    private readonly int? num_embeds_ada_norm;
    private readonly bool use_linear_projection;
    private readonly bool only_cross_attention;
    private readonly bool double_self_attention;
    private readonly bool upcast_attention;
    private readonly string norm_type;
    private readonly bool norm_elementwise_affine;
    private readonly double norm_eps;
    private readonly string attention_type;
    private readonly int? caption_channels;
    private readonly double? interpolation_scale;
    private readonly bool use_additional_conditions = false;

    private readonly bool is_input_continuous;
    private readonly bool is_input_vectorized;
    private readonly bool is_input_patches;

    private readonly GroupNorm norm;
    private readonly Module<Tensor, Tensor> proj_in;
    private readonly ModuleList<BasicTransformerBlock> transformer_blocks;
    private readonly Module<Tensor, Tensor> proj_out;

    public Transformer2DModel(
        int num_attention_heads = 16,
        int attention_head_dim = 88,
        int? in_channels = null,
        int? out_channels = null,
        int num_layers = 1,
        double dropout = 0.0,
        int norm_num_groups = 32,
        int? cross_attention_dim = null,
        bool attention_bias = false,
        int? sample_size = null,
        int? num_vector_embeds = null,
        int? patch_size = null,
        string activation_fn = "geglu",
        int? num_embeds_ada_norm = null,
        bool use_linear_projection = false,
        bool only_cross_attention = false,
        bool double_self_attention = false,
        bool upcast_attention = false,
        string norm_type = "layer_norm",
        bool norm_elementwise_affine = true,
        double norm_eps = 1e-5,
        string attention_type = "default",
        int? caption_channels = null,
        double? interpolation_scale = null)
        : base("Transformer2DModel")
    {
        this.num_attention_heads = num_attention_heads;
        this.attention_head_dim = attention_head_dim;
        this.in_channels = in_channels;
        this.out_channels = out_channels ?? in_channels;
        this.num_layers = num_layers;
        this.dropout = dropout;
        this.norm_num_groups = norm_num_groups;
        this.cross_attention_dim = cross_attention_dim;
        this.attention_bias = attention_bias;
        this.sample_size = sample_size;
        this.num_vector_embeds = num_vector_embeds;
        this.patch_size = patch_size;
        this.activation_fn = activation_fn;
        this.num_embeds_ada_norm = num_embeds_ada_norm;
        this.use_linear_projection = use_linear_projection;
        this.only_cross_attention = only_cross_attention;
        this.double_self_attention = double_self_attention;
        this.upcast_attention = upcast_attention;
        this.norm_type = norm_type;
        this.norm_elementwise_affine = norm_elementwise_affine;
        this.norm_eps = norm_eps;
        this.attention_type = attention_type;
        this.caption_channels = caption_channels;
        this.interpolation_scale = interpolation_scale;

        if (norm_type != "layer_norm")
        {
            throw new NotImplementedException("Only layer_norm is supported for now");
        }

        if (patch_size is not null)
        {
            throw new ArgumentNullException("patch_size");
        }

        if (num_embeds_ada_norm is not null)
        {
            throw new ArgumentNullException("num_embeds_ada_norm");
        }
        var inner_dim = attention_head_dim * num_attention_heads;
        this.is_input_continuous = (in_channels is not null) && (patch_size is null);
        this.is_input_vectorized = false;
        this.is_input_patches = false;

        if (this.is_input_continuous)
        {
            this.norm = GroupNorm(num_groups: norm_num_groups, num_channels: in_channels!.Value, eps: 1e-6, affine: true);

            if (this.use_linear_projection)
            {
                this.proj_in = Linear(in_channels!.Value, inner_dim);
            }
            else
            {
                this.proj_in = Conv2d(in_channels!.Value, inner_dim, kernelSize: 1, stride: 1, padding: Padding.Valid);
            }
        }

        this.transformer_blocks = new ModuleList<BasicTransformerBlock>();
        for (int i = 0; i < num_layers; i++)
        {
            this.transformer_blocks.Add(new BasicTransformerBlock(
                dim: inner_dim,
                num_attention_heads: num_attention_heads,
                attention_head_dim: attention_head_dim,
                dropout: dropout,
                cross_attention_dim: cross_attention_dim,
                activation_fn: activation_fn,
                num_embeds_ada_norm: num_embeds_ada_norm,
                attention_bias: attention_bias,
                only_cross_attention: only_cross_attention,
                double_self_attention: double_self_attention,
                upcast_attention: upcast_attention,
                norm_type: norm_type,
                norm_elementwise_affine: norm_elementwise_affine,
                norm_eps: norm_eps,
                attention_type: attention_type
            ));
        }

        // 4. Define output layers
        if (this.is_input_continuous)
        {
            if (this.use_linear_projection)
            {
                this.proj_out = Linear(inner_dim, in_channels!.Value);
            }
            else
            {
                this.proj_out = Conv2d(inner_dim, in_channels!.Value, kernelSize: 1, stride: 1, padding: Padding.Valid);
            }
        }
    }

    public override Transformer2DModelOutput forward(
        Tensor hidden_states,
        Tensor? encoder_hidden_states = null,
        Tensor? timestep = null,
        Tensor? class_labels = null,
        Tensor? attention_mask = null,
        Tensor? encoder_attention_mask = null)
    {
        if (attention_mask is not null && attention_mask.ndim == 2)
        {
            // assume that mask is expressed as:
            //   (1 = keep,      0 = discard)
            // convert mask into a bias that can be added to attention scores:
            //       (keep = +0,     discard = -10000.0)
            attention_mask = (1-attention_mask.to(hidden_states.dtype)) * -10000.0;
            attention_mask = attention_mask.unsqueeze(1);
        }

        // convert encoder_attention_mask to a bias the same way we do for attention_mask
        if (encoder_attention_mask is not null && encoder_attention_mask.ndim == 2)
        {
            encoder_attention_mask = (1-encoder_attention_mask.to(hidden_states.dtype)) * -10000.0;
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1);
        }
        var residual = hidden_states;
        var batch = hidden_states.shape[0];
        var inner_dim = hidden_states.shape[1];
        var height = hidden_states.shape[2];
        var width = hidden_states.shape[3];

        if (this.is_input_continuous)
        {
            hidden_states = this.norm.forward(hidden_states);
            if (this.use_linear_projection)
            {
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim);
                hidden_states = this.proj_in.forward(hidden_states);
            }
            else
            {
                hidden_states = this.proj_in.forward(hidden_states);
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim);
            }
        }

        // 2. Blocks
        foreach (var block in this.transformer_blocks)
        {
            hidden_states = block.forward(
                hidden_states,
                attention_mask: attention_mask,
                encoder_hidden_states: encoder_hidden_states,
                encoder_attention_mask: encoder_attention_mask,
                timestep: timestep);
        }

        // 3. Output
        if (this.is_input_continuous)
        {
            if (this.use_linear_projection)
            {
                hidden_states = this.proj_out.forward(hidden_states);
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous();
            }
            else
            {
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous();
                hidden_states = this.proj_out.forward(hidden_states);
            }

            hidden_states = hidden_states + residual;
        }

        return new Transformer2DModelOutput(hidden_states);
    }
}

public class Transformer2DModelOutput
{
    public Transformer2DModelOutput(Tensor sample)
    {
        Sample = sample;
    }
    public Tensor Sample { get; }
}