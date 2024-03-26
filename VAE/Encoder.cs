using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

/// <summary>
/// The Encoder layer of a variational autoencoder that compresses the input data into a latent space.
/// </summary>
public class Encoder : Module<Tensor, Tensor>
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int[] _blockOutChannels;
    private readonly string[] _downBlockTypes;
    private readonly int _layersPerBlock;
    private readonly int _normNumGroups;
    private readonly string _activationFunction;
    private readonly bool gradient_checkpointing = false;
    private readonly Module<Tensor, Tensor> conv_in;
    private readonly ModuleList<Module<Tensor, Tensor>> down_blocks;
    private readonly UNetMidBlock2D mid_block;
    private readonly Module<Tensor, Tensor> conv_out;
    private readonly Module<Tensor, Tensor> conv_norm_out;
    private readonly Module<Tensor, Tensor> conv_act;


    /// <summary>
    /// Initializes a new instance of the <see cref="Encoder"/> class.
    /// </summary>
    /// <param name="inChannels">The number of input channels.</param>
    /// <param name="latentChannels">The number of latent channels.</param>
    /// <param name="blockOutChannels">The number of output channels for each block.</param>
    /// <param name="downBlockTypes">The types of blocks to use for downscaling.</param>
    /// <param name="layersPerBlock">The number of layers per block.</param>
    /// <param name="normNumGroups">The number of groups for normalization.</param>
    /// <param name="activationFunction">The activation function to use.</param>
    public Encoder(
        int? inChannels = null,
        int? outChannels = null,
        int[]? blockOutChannels = null,
        string[]? downBlockTypes = null,
        int layersPerBlock = 2,
        int normNumGroups = 32,
        string activationFunction = "silu",
        bool doubleZ = true,
        bool midBlockAddAttention = true)
        : base(nameof(Encoder))
    {
        _inChannels = inChannels ?? 3;
        _outChannels = outChannels ?? 3;
        _blockOutChannels = blockOutChannels ?? [64];
        _downBlockTypes = downBlockTypes ?? ["DownEncoderBlock2D"];
        _layersPerBlock = layersPerBlock;
        _normNumGroups = normNumGroups;
        _activationFunction = activationFunction;

        this.conv_in = torch.nn.Conv2d(this._inChannels, this._blockOutChannels[0], kernelSize: 3, stride: 1, padding: 1);
        this.down_blocks = new ModuleList<Module<Tensor, Tensor>>();

        var output_channel = _blockOutChannels[0];
        for (int i = 0; i < _blockOutChannels.Length; i++)
        {
            var input_channel = output_channel;
            output_channel = _blockOutChannels[i];
            var is_final_block = i == _blockOutChannels.Length - 1;
            var down_block = new DownEncoderBlock2D(
                in_channels: input_channel,
                out_channels: output_channel,
                add_downsample: !is_final_block,
                num_layers: _layersPerBlock,
                resnet_act_fun: _activationFunction,
                resnet_groups: _normNumGroups,
                downsample_padding: 0);

            this.down_blocks.Add(down_block);
        }

        // mid
        this.mid_block = new UNetMidBlock2D(
            in_channels: _blockOutChannels[^1],
            resnet_eps: 1e-6f,
            resnet_act_fn: activationFunction,
            output_scale_factor: 1,
            resnet_time_scale_shift: "default",
            attention_head_dim: _blockOutChannels[^1],
            resnet_groups: normNumGroups,
            add_attention: midBlockAddAttention);

        // out
        this.conv_norm_out = nn.GroupNorm(num_groups: normNumGroups, num_channels: _blockOutChannels[^1], eps: 1e-6f);
        this.conv_act = nn.SiLU();
        var conv_out_channels = doubleZ ? _outChannels * 2 : _outChannels;
        this.conv_out = nn.Conv2d(_blockOutChannels[^1], conv_out_channels, kernelSize: 3, padding: Padding.Same);
    }

    public override Tensor forward(Tensor sample)
    {
        sample = this.conv_in.forward(sample);
        // down
        foreach (var down_block in this.down_blocks)
        {
            sample = down_block.forward(sample);
        }

        // mid
        sample = this.mid_block.forward(sample);
        // post-process
        sample = this.conv_norm_out.forward(sample);
        sample = this.conv_act.forward(sample);
        sample = this.conv_out.forward(sample);

        return sample;
    }
}