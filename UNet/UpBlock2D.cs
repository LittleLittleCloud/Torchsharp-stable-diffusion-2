namespace SD;

public class UpBlock2D : Module<UpBlock2DInput, Tensor>
{
    public UpBlock2D(
        int in_channels,
        int prev_output_channel,
        int out_channels,
        int temb_channels,
        int? resolution_idx = null,
        float dropout = 0.0f,
        int num_layers = 1,
        float resnet_eps = 1e-6f,
        string resnet_time_scale_shift = "default",
        string resnet_act_fn = "swish",
        int? resnet_groups = 32,
        bool resnet_pre_norm = true,
        float output_scale_factor = 1.0f,
        bool add_upsample = true
    ): base(nameof(UpBlock2D))
    {
        var resnets = new ModuleList<ResnetBlock2D>();
        for(int i = 0; i != num_layers; ++i)
        {
            var res_skip_channels = i == num_layers - 1 ? in_channels : out_channels;
            var resnet_in_channels = i == 0 ? prev_output_channel : out_channels;

            resnets.Add(
                new ResnetBlock2D(
                    in_channels: resnet_in_channels + res_skip_channels,
                    out_channels: out_channels,
                    temb_channels: temb_channels,
                    eps: resnet_eps,
                    groups: resnet_groups ?? 32,
                    dropout: dropout,
                    time_embedding_norm: resnet_time_scale_shift,
                    non_linearity: resnet_act_fn,
                    output_scale_factor: output_scale_factor,
                    pre_norm: resnet_pre_norm)
            );
        }

        this.resnets = resnets;

        if (add_upsample)
        {
            this.upsamplers = new ModuleList<Upsample2D>();
            this.upsamplers.Add(new Upsample2D(
                channels: out_channels,
                use_conv: true,
                out_channels: out_channels
            ));
        }

        this.resolution_idx = resolution_idx;
    }

    private readonly ModuleList<ResnetBlock2D> resnets;
    private readonly ModuleList<Upsample2D>? upsamplers;
    private readonly int? resolution_idx;
    public ModuleList<ResnetBlock2D> Resnets => resnets;

    public override Tensor forward(UpBlock2DInput x)
    {
        var hidden_states = x.HiddenStates;
        foreach (var resnet in resnets)
        {
            var res_hidden_states = x.ResHiddenStatesTuple[^1];
            var res_hidden_states_tuple = x.ResHiddenStatesTuple[..^1];

            hidden_states = torch.cat(new Tensor[] {hidden_states, res_hidden_states}, 1);
            hidden_states = resnet.forward(hidden_states, x.Temb);
        }

        if (upsamplers is not null)
        {
            foreach (var upsample in upsamplers)
            {
                hidden_states = upsample.forward(hidden_states, x.UpsampleSize);
            }
        }

        return hidden_states;
    }
}