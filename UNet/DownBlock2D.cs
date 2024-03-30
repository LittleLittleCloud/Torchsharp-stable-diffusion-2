namespace SD;

public class DownBlock2DInput
{
    public Tensor HiddenStates {get;}
    public Tensor? Temb {get;} = null;

    public Tensor? EncoderHiddenStates {get;} = null;
    public Tensor? AttentionMask {get;} = null;
    public Tensor? EncoderAttentionMask {get;} = null;
    public Tensor? AdditionalResiduals {get;} = null;

    public DownBlock2DInput(
        Tensor hiddenStates,
        Tensor? temb = null,
        Tensor? encoderHiddenStates = null,
        Tensor? attentionMask = null,
        Tensor? encoderAttentionMask = null,
        Tensor? additionalResiduals = null)
    {
        HiddenStates = hiddenStates;
        Temb = temb;
        EncoderHiddenStates = encoderHiddenStates;
        AttentionMask = attentionMask;
        EncoderAttentionMask = encoderAttentionMask;
        AdditionalResiduals = additionalResiduals;
    }
}

public class DownBlock2DOutput
{
    public Tensor HiddenStates {get;}
    public Tensor[]? OutputStates {get;} = null;

    public DownBlock2DOutput(Tensor hiddenStates, Tensor[]? outputStates = null)
    {
        HiddenStates = hiddenStates;
        OutputStates = outputStates;
    }
}

public class DownBlock2D: Module<DownBlock2DInput, DownBlock2DOutput>
{
    private ModuleList<ResnetBlock2D> resnets;
    private ModuleList<Module<Tensor, Tensor>>? attentions;
    private ModuleList<Module<Tensor, Tensor>>? downsamplers;

    public DownBlock2D(
        int in_channels,
        int out_channels,
        int temb_channels,
        float dropout = 0.0f,
        int num_layers = 1,
        float resnet_eps = 1e-6f,
        string resnet_time_scale_shift = "default",
        string resnet_act_fn = "swish",
        int? resnet_groups = 32,
        bool resnet_pre_norm = true,
        float output_scale_factor = 1.0f,
        bool? add_downsample = true,
        int? downsample_padding = 1)
        : base(nameof(DownBlock2D))
    {
        var resnets = new ModuleList<ResnetBlock2D>();
        for (int i = 0; i < num_layers; i++)
        {
            in_channels = i == 0 ? in_channels : out_channels;
            resnets.Add(new ResnetBlock2D(
                in_channels: in_channels,
                out_channels: out_channels,
                temb_channels: temb_channels,
                eps: resnet_eps,
                groups: resnet_groups ?? throw new ArgumentNullException(nameof(resnet_groups)),
                dropout: dropout,
                time_embedding_norm: resnet_time_scale_shift,
                non_linearity: resnet_act_fn,
                output_scale_factor: output_scale_factor,
                pre_norm: resnet_pre_norm));
        }

        this.resnets = resnets;

        if (add_downsample is true)
        {
            this.downsamplers = new ModuleList<Module<Tensor, Tensor>>();
            this.downsamplers.Add(new Downsample2D(
                channels: out_channels,
                use_conv: true,
                out_channels: out_channels,
                name: "op",
                padding: downsample_padding));
        }
    }

    public override DownBlock2DOutput forward(DownBlock2DInput input)
    {
        var hiddenStates = input.HiddenStates;
        var temb = input.Temb;
        var encoderHiddenStates = input.EncoderHiddenStates;
        var attentionMask = input.AttentionMask;
        var encoderAttentionMask = input.EncoderAttentionMask;
        var additionalResiduals = input.AdditionalResiduals;

        var output_states = new List<Tensor>();

        foreach (var resnet in resnets)
        {
            hiddenStates = resnet.forward(hiddenStates, temb);
            output_states.Add(hiddenStates);
        }

        if (downsamplers != null)
        {
            foreach (var downsample in downsamplers)
            {
                hiddenStates = downsample.forward(hiddenStates);
            }

            output_states.Add(hiddenStates);
        }

        return new DownBlock2DOutput(hiddenStates, output_states.ToArray());
    }
}