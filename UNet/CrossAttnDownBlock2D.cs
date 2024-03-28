namespace SD;
public class CrossAttnDownBlock2D : Module<Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?, (Tensor, Tensor[])>
{
    private readonly bool has_cross_attention;
    private readonly int num_attention_heads;

    private readonly ModuleList<ResnetBlock2D> resnets;
    private readonly ModuleList<Module> attentions;
    private readonly ModuleList<Downsample2D>? downsamplers = null;

    public CrossAttnDownBlock2D(
        int in_channels,
        int out_channels,
        int temb_channels,
        double dropout = 0.0,
        int num_layers = 1,
        int[]? transformer_layers_per_block = null,
        double resnet_eps = 1e-6,
        string resnet_time_scale_shift = "default",
        string resnet_act_fn = "swish",
        int resnet_groups = 32,
        bool resnet_pre_norm = true,
        int num_attention_heads = 1,
        int cross_attention_dim = 1280,
        double output_scale_factor = 1.0,
        int downsample_padding = 1,
        bool add_downsample = true,
        bool dual_cross_attention = false,
        bool use_linear_projection = false,
        bool only_cross_attention = false,
        bool upcast_attention = false,
        string attention_type = "default"
    ): base(nameof(CrossAttnDownBlock2D))
    {
        ModuleList<ResnetBlock2D> resnets = new ModuleList<ResnetBlock2D>();
        ModuleList<Module> attentions = new ModuleList<Module>();

        this.has_cross_attention = true;
        this.num_attention_heads = num_attention_heads;
        transformer_layers_per_block = transformer_layers_per_block ?? Enumerable.Repeat(num_layers, 1).ToArray();

        for(int i = 0; i != num_layers; ++i)
        {
            in_channels = i == 0 ? in_channels : out_channels;
            resnets.Add(
                new ResnetBlock2D(
                    in_channels: in_channels,
                    out_channels: out_channels,
                    temb_channels: temb_channels,
                    eps: (float)resnet_eps,
                    groups: resnet_groups,
                    dropout: (float)dropout,
                    time_embedding_norm: resnet_time_scale_shift,
                    non_linearity: resnet_act_fn,
                    output_scale_factor: (float)output_scale_factor,
                    pre_norm: resnet_pre_norm));

            if (!dual_cross_attention)
            {
                attentions.Add(
                    new Transformer2DModel(
                        num_attention_heads: num_attention_heads,
                        out_channels: out_channels / num_attention_heads,
                        in_channels: out_channels,
                        num_layers: transformer_layers_per_block[i],
                        cross_attention_dim: cross_attention_dim,
                        norm_num_groups: resnet_groups,
                        use_linear_projection: use_linear_projection,
                        only_cross_attention: only_cross_attention,
                        upcast_attention: upcast_attention,
                        attention_type: attention_type));
            }
            else
            {
                attentions.Add(
                    new DualTransformer2DModel(
                        num_attention_heads: num_attention_heads,
                        attention_head_dim: out_channels / num_attention_heads,
                        in_channels: out_channels,
                        num_layers: 1,
                        cross_attention_dim: cross_attention_dim,
                        norm_num_groups: resnet_groups));
            }
        }

        this.resnets = resnets;
        this.attentions = attentions;

        if (add_downsample)
        {
            this.downsamplers = new ModuleList<Downsample2D>();
            this.downsamplers.Add(
                new Downsample2D(
                    channels: out_channels,
                    use_conv: true,
                    out_channels: out_channels,
                    padding: downsample_padding,
                    name: "op"));
        }
    }

    public override (Tensor, Tensor[]) forward(
        Tensor hidden_states,
        Tensor? temb = null,
        Tensor? encoder_hidden_states = null,
        Tensor? attention_mask = null,
        Tensor? encoder_attention_mask = null,
        Tensor? additional_residuals = null)
    {
        List<Tensor> output_states = new List<Tensor>();

        var blocks = this.resnets.Zip(this.attentions, (resnet, attention) => (resnet, attention)).ToArray();

        for(int i = 0; i !=blocks.Count(); ++i)
        {
            var (resnet, attention) = blocks[i];
            hidden_states = resnet.forward(hidden_states, temb);
            if (attention is Transformer2DModel transformer)
            {
                hidden_states = transformer.forward(
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    encoder_attention_mask,
                    additional_residuals).Sample;
            }
            else if (attention is DualTransformer2DModel dual_transformer)
            {
                hidden_states = dual_transformer.forward(
                    hidden_states,
                    encoder_hidden_states ?? throw new ArgumentNullException(nameof(encoder_hidden_states)),
                    attention_mask: attention_mask).Sample;
            }
            else
            {
                throw new NotImplementedException();
            }

            if (i == blocks.Count() - 1 && additional_residuals is not null)
            {
                hidden_states = hidden_states + additional_residuals;
            }

            output_states.Add(hidden_states);
        }

        if (this.downsamplers is not null)
        {
            foreach (var downsample in this.downsamplers)
            {
                hidden_states = downsample.forward(hidden_states);
            }

            output_states.Add(hidden_states);
        }

        return (hidden_states, output_states.ToArray());
    }
    

}