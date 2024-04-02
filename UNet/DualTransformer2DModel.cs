public class DualTransformer2DModel : Module<Tensor, Tensor, Tensor?, Tensor?, Transformer2DModelOutput>
{
    private readonly float mix_ratio = 0.5f;
    private readonly int[] condition_lengths = [77, 257];
    private readonly int[] transformer_index_for_condition = [1, 0];
    private readonly ModuleList<Transformer2DModel> transformers;

    public DualTransformer2DModel(
        int num_attention_heads = 16,
        int attention_head_dim = 88,
        int? in_channels = null,
        int num_layers = 1,
        double dropout = 0.0,
        int norm_num_groups = 32,
        int? cross_attention_dim = null,
        bool attention_bias = false,
        int? sample_size = null,
        int? num_vector_embeds = null,
        string activation_fn = "geglu",
        int? num_embeds_ada_norm = null
    ) : base(nameof(DualTransformer2DModel))
    {
        this.transformers = new ModuleList<Transformer2DModel>();
        for(int i = 0; i < 2; i++)
        {
            transformers.Add(new Transformer2DModel(
                num_attention_heads: num_attention_heads,
                attention_head_dim: attention_head_dim,
                in_channels: in_channels,
                num_layers: num_layers,
                dropout: dropout,
                norm_num_groups: norm_num_groups,
                cross_attention_dim: cross_attention_dim,
                attention_bias: attention_bias,
                sample_size: sample_size,
                num_vector_embeds: num_vector_embeds,
                activation_fn: activation_fn,
                num_embeds_ada_norm: num_embeds_ada_norm));
        }
    }

    public override Transformer2DModelOutput forward(
        Tensor hidden_states,
        Tensor encoder_hidden_states,
        Tensor? timestep = null,
        Tensor? attention_mask = null)
    {
        var input_states = hidden_states;
        List<Tensor> encoded_states = [];
        var tokens_start = 0;

        for(int i = 0; i < 2; i++)
        {
            var condition_state = encoder_hidden_states[.., tokens_start..(tokens_start + condition_lengths[i])];
            var transformer_index = transformer_index_for_condition[i];
            var transformer = transformers[transformer_index];
            var encoded_state = transformer.forward(
                input_states, condition_state, timestep);
            encoded_states.Add(encoded_state.Sample - input_states);
            tokens_start += condition_lengths[i];
        }

        var output_states = encoded_states[0] * this.mix_ratio + encoded_states[1] * (1 - this.mix_ratio);
        output_states = input_states + output_states;

        return new Transformer2DModelOutput(output_states);
    }
}