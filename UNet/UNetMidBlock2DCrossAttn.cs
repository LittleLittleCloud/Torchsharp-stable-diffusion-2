namespace SD;

public class UNetMidBlock2DCrossAttnInput
{
    public UNetMidBlock2DCrossAttnInput(
        Tensor hiddenStates,
        Tensor? temb = null,
        Tensor? encoderHiddenStates = null,
        Tensor? attentionMask = null,
        Dictionary<string, object>? crossAttentionKwargs = null,
        Tensor? encoderAttentionMask = null)
    {
        HiddenStates = hiddenStates;
        Temb = temb;
        EncoderHiddenStates = encoderHiddenStates;
        AttentionMask = attentionMask;
        CrossAttentionKwargs = crossAttentionKwargs;
        EncoderAttentionMask = encoderAttentionMask;
    }

    public Tensor HiddenStates { get; }

    public Tensor? Temb { get; }

    public Tensor? EncoderHiddenStates { get; }

    public Tensor? AttentionMask { get; }

    public Dictionary<string, object>? CrossAttentionKwargs { get; }

    public Tensor? EncoderAttentionMask { get; }
}

public class UNetMidBlock2DCrossAttn: Module<UNetMidBlock2DCrossAttnInput, Tensor>
{
    private readonly bool has_cross_attention;
    private readonly int num_attention_heads;
    private readonly ModuleList<ResnetBlock2D> resnets;
    private readonly ModuleList<Module> attentions;

    public UNetMidBlock2DCrossAttn(
        int in_channels,
        int temb_channels,
        float dropout = 0.0f,
        int num_layers = 1,
        int[]? transformer_layers_per_block = null,
        float resnet_eps = 1e-6f,
        string resnet_time_scale_shift = "default",
        string resnet_act_fn = "swish",
        int resnet_groups = 32,
        bool resnet_pre_norm = true,
        int num_attention_heads = 1,
        float output_scale_factor = 1.0f,
        int cross_attention_dim = 1280,
        bool dual_cross_attention = false,
        bool use_linear_projection = false,
        bool upcast_attention = false,
        string attention_type = "default"
    ): base(nameof(UNetMidBlock2DCrossAttn))
    {
        ModuleList<ResnetBlock2D> resnets = new ModuleList<ResnetBlock2D>();
        ModuleList<Module> attentions = new ModuleList<Module>();

        this.has_cross_attention = true;
        this.num_attention_heads = num_attention_heads;
        transformer_layers_per_block = transformer_layers_per_block ?? Enumerable.Repeat(num_layers, 1).ToArray();

        resnets.Add(
            new ResnetBlock2D(
                in_channels: in_channels,
                out_channels: in_channels,
                temb_channels: temb_channels,
                eps: (float)resnet_eps,
                groups: resnet_groups,
                dropout: (float)dropout,
                time_embedding_norm: resnet_time_scale_shift,
                non_linearity: resnet_act_fn,
                output_scale_factor: (float)output_scale_factor,
                pre_norm: resnet_pre_norm));
        for(int i = 0; i != num_layers; ++i)
        {
            resnets.Add(
                new ResnetBlock2D(
                    in_channels: in_channels,
                    out_channels: in_channels,
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
                        attention_head_dim: in_channels / num_attention_heads,
                        in_channels: in_channels,
                        num_layers: transformer_layers_per_block[i],
                        cross_attention_dim: cross_attention_dim,
                        norm_num_groups: resnet_groups,
                        use_linear_projection: use_linear_projection,
                        upcast_attention: upcast_attention,
                        attention_type: attention_type));
            }
            else
            {
                attentions.Add(
                    new DualTransformer2DModel(
                        num_attention_heads: num_attention_heads,
                        attention_head_dim: in_channels / num_attention_heads,
                        in_channels: in_channels,
                        num_layers: 1,
                        cross_attention_dim: cross_attention_dim,
                        norm_num_groups: resnet_groups));
            }
        }

        this.resnets = resnets;
        this.attentions = attentions;
    }

    public override Tensor forward(UNetMidBlock2DCrossAttnInput input)
    {
        var hiddenStates = input.HiddenStates;
        var temb = input.Temb;
        var encoderHiddenStates = input.EncoderHiddenStates;
        var attentionMask = input.AttentionMask;
        var crossAttentionKwargs = input.CrossAttentionKwargs;
        var encoderAttentionMask = input.EncoderAttentionMask;

        hiddenStates = this.resnets[0].forward(hiddenStates, temb);

        foreach (var (resnet, attention) in this.resnets.Skip(1).Zip(this.attentions))
        {
            if (attention is Transformer2DModel transformer)
            {
                hiddenStates = transformer.forward(
                    hiddenStates,
                    encoder_hidden_states: encoderHiddenStates,
                    attention_mask: attentionMask,
                    encoder_attention_mask: encoderAttentionMask).Sample;
            }
            else if (attention is DualTransformer2DModel dualTransformer)
            {
                hiddenStates = dualTransformer.forward(
                    hiddenStates,
                    encoder_hidden_states: encoderHiddenStates ?? throw new ArgumentNullException(nameof(encoderHiddenStates)),
                    attention_mask: attentionMask).Sample;
            }

            hiddenStates = resnet.forward(hiddenStates, temb);
        }

        return hiddenStates;
    }
}