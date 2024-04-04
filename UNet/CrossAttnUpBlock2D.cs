
namespace SD;
public class UpBlock2DInput
{
    public UpBlock2DInput(
        Tensor hiddenStates,
        Tensor[] resHiddenStatesTuple, 
        Tensor? temb = null,
        Tensor? encoderHiddenStates = null,
        Dictionary<string, object>? crossAttentionKwargs = null,
        int? upsampleSize = null,
        int[]? upsampleSizeList = null,
        Tensor? attentionMask = null,
        Tensor? encoderAttentionMask = null)
    {
        HiddenStates = hiddenStates;
        ResHiddenStatesTuple = resHiddenStatesTuple;
        Temb = temb;
        EncoderHiddenStates = encoderHiddenStates;
        CrossAttentionKwargs = crossAttentionKwargs;
        UpsampleSize = upsampleSize;
        AttentionMask = attentionMask;
        UpsampleSizeList = upsampleSizeList;
        EncoderAttentionMask = encoderAttentionMask;
    }
    public Tensor HiddenStates { get; }
    public Tensor[] ResHiddenStatesTuple { get; }
    public Tensor? Temb { get; }
    public Tensor? EncoderHiddenStates { get; }
    public Dictionary<string, object>? CrossAttentionKwargs { get; }

    public int? UpsampleSize { get; }

    public int[]? UpsampleSizeList { get; }

    public Tensor? AttentionMask { get; }

    public Tensor? EncoderAttentionMask { get; }

}

public class CrossAttnUpBlock2D : Module<UpBlock2DInput, Tensor>
{
    private readonly bool has_cross_attention;
    private readonly int num_attention_heads;
    private readonly ModuleList<ResnetBlock2D> resnets;
    private readonly ModuleList<Module> attentions;
    private readonly ModuleList<Upsample2D>? upsamplers = null;
    private readonly int? resolution_idx;
    public CrossAttnUpBlock2D(
        int in_channels,
        int out_channels,
        int prev_output_channel,
        int temb_channels,
        int? resolution_idx = null,
        float dropout = 0.0f,
        int num_layers = 1,
        int[]? transformer_layers_per_block = null,
        float resnet_eps = 1e-6f,
        string resnet_time_scale_shift = "default",
        string resnet_act_fn = "swish",
        int resnet_groups = 32,
        bool resnet_pre_norm = true,
        int num_attention_heads = 1,
        int cross_attention_dim = 1280,
        float output_scale_factor = 1.0f,
        bool add_upsample = true,
        bool dual_cross_attention = false,
        bool use_linear_projection = false,
        bool only_cross_attention = false,
        bool upcast_attention = false,
        string attention_type = "default",
        ScalarType dtype = ScalarType.Float32)
        : base(nameof(CrossAttnUpBlock2D))
    {
        ModuleList<ResnetBlock2D> resnets = new ModuleList<ResnetBlock2D>();
        ModuleList<Module> attentions = new ModuleList<Module>();

        this.has_cross_attention = true;
        this.num_attention_heads = num_attention_heads;
        transformer_layers_per_block = transformer_layers_per_block ?? Enumerable.Repeat(1, num_layers).ToArray();

        for (int i = 0; i != num_layers; ++i)
        {
            var res_skip_channels = i == num_layers - 1 ? in_channels : out_channels;
            var resnet_in_channels = i == 0 ? prev_output_channel : out_channels;

            resnets.Add(
                new ResnetBlock2D(
                    in_channels: resnet_in_channels + res_skip_channels,
                    out_channels: out_channels,
                    temb_channels: temb_channels,
                    eps: resnet_eps,
                    groups: resnet_groups,
                    dropout: dropout,
                    time_embedding_norm: resnet_time_scale_shift,
                    non_linearity: resnet_act_fn,
                    output_scale_factor: output_scale_factor,
                    pre_norm: resnet_pre_norm,
                    dtype: dtype));
            
            if (!dual_cross_attention)
            {
                attentions.Add(
                    new Transformer2DModel(
                        num_attention_heads: num_attention_heads,
                        attention_head_dim: out_channels / num_attention_heads,
                        in_channels: out_channels,
                        num_layers: transformer_layers_per_block[i],
                        cross_attention_dim: cross_attention_dim,
                        norm_num_groups: resnet_groups,
                        use_linear_projection: use_linear_projection,
                        only_cross_attention: only_cross_attention,
                        upcast_attention: upcast_attention,
                        attention_type: attention_type,
                        dtype: dtype));
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
                        norm_num_groups: resnet_groups,
                        dtype: dtype));
            
            }
        }
        
        this.resnets = resnets;
        this.attentions = attentions;

        if (add_upsample)
        {
            this.upsamplers = new ModuleList<Upsample2D>();
            this.upsamplers.Add(
                new Upsample2D(
                    channels: out_channels,
                    use_conv: true,
                    out_channels: out_channels,
                    dtype: dtype));
        }
        this.resolution_idx = resolution_idx;
    }

    public ModuleList<ResnetBlock2D> Resnets => resnets;
    public override Tensor forward(UpBlock2DInput input)
    {
        var hiddenStates = input.HiddenStates;
        var resHiddenStatesTuple = input.ResHiddenStatesTuple;
        var temb = input.Temb;
        var encoderHiddenStates = input.EncoderHiddenStates;
        var crossAttentionKwargs = input.CrossAttentionKwargs;
        var upsampleSize = input.UpsampleSize;
        var attentionMask = input.AttentionMask;
        var encoderAttentionMask = input.EncoderAttentionMask;

        foreach(var (resnet, attention) in resnets.Zip(attentions))
        {
            // pop res hidden states
            var res_hidden_states = resHiddenStatesTuple[^1];
            resHiddenStatesTuple = resHiddenStatesTuple[..^1];

            hiddenStates = torch.cat([hiddenStates, res_hidden_states], 1);
            hiddenStates = resnet.forward(hiddenStates, temb);
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
        }

        if (upsamplers != null)
        {
            foreach(var upsample in upsamplers)
            {
                hiddenStates = upsample.forward(hiddenStates, upsampleSize);
            }
        }

        return hiddenStates;
    }
}
