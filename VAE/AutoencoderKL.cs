using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using System.Text.Json;
using TorchSharp.PyBridge;

namespace SD;

public class AutoencoderKL : Module<Tensor, bool, Generator?, Tensor>, IModelConfigLoader<AutoencoderKL>
{
    private readonly int in_channels;
    private readonly int out_channels;
    private readonly string[] down_block_types;
    private readonly string[] up_block_types;
    private readonly int[] block_out_channels;
    private readonly int layers_per_block;
    private readonly string act_fn;

    private readonly int latent_channels;
    private readonly int norm_num_groups;
    private readonly int sample_size;
    private readonly float scaling_factor;
    private readonly float[]? latents_mean;
    private readonly float[]? latents_std;
    private readonly bool force_upcast;

    private readonly Encoder encoder;

    private readonly Decoder decoder;

    private readonly Conv2d quant_conv;
    private readonly Conv2d post_quant_conv;
    private readonly ScalarType dtype;

    /// <summary>
    /// Create an AutoencoderKL model.
    /// </summary>
    /// <param name="config">AutoencoderKL config</param>
    /// <param name="dtype">the default dtype to use</param>
    public AutoencoderKL(
        Config config,
        ScalarType dtype = ScalarType.Float32)
        : base(nameof(AutoencoderKL))
    {
        this.in_channels = config!.InChannels;
        this.out_channels = config!.OutChannels;
        this.down_block_types = config!.DownBlockTypes;
        this.up_block_types = config!.UpBlockTypes;
        this.block_out_channels = config!.BlockOutChannels;
        this.layers_per_block = config!.LayersPerBlock;
        this.act_fn = config!.ActivationFunction;
        this.latent_channels = config!.LatentChannels;
        this.norm_num_groups = config!.NormNumGroups;
        this.sample_size = config!.SampleSize;
        this.scaling_factor = config!.ScalingFactor;
        this.latents_mean = config!.LatentsMean;
        this.latents_std = config!.LatentsStd;
        this.force_upcast = config!.ForceUpcast;
        this.dtype = dtype;

        

        this.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernelSize: 1, padding: Padding.Valid, dtype: this.dtype);
        this.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernelSize: 1, padding: Padding.Same, dtype: this.dtype);

        this.Config = config;

        this.encoder = new Encoder(
            inChannels: in_channels,
            outChannels: latent_channels,
            downBlockTypes: down_block_types,
            blockOutChannels: block_out_channels,
            layersPerBlock: layers_per_block,
            activationFunction: act_fn,
            mid_block_from_deprecated_attn_block: config.DecoderMidBlockFromDeprecatedAttnBlock,
            normNumGroups: norm_num_groups,
            doubleZ: true,
            dtype: this.dtype);

        this.decoder = new Decoder(
            in_channels: latent_channels,
            out_channels: out_channels,
            up_block_types: up_block_types,
            block_out_channels: block_out_channels,
            layers_per_block: layers_per_block,
            norm_num_groups: norm_num_groups,
            act_fn: act_fn,
            mid_block_from_deprecated_attn_block: config.DecoderMidBlockFromDeprecatedAttnBlock,
            mid_block_add_attention: true,
            dtype: this.dtype);

        RegisterComponents();
    }

    public Decoder Decoder => this.decoder;

    public Encoder Encoder => this.encoder;

    public Config Config {get;}

    public DiagonalGaussianDistribution encode(Tensor x)
    {
        var h = this.encoder.forward(x);
        var moments = this.quant_conv.forward(h);
        var posterior = new DiagonalGaussianDistribution(moments);

        return posterior;
    }

    public Tensor _decode(Tensor z)
    {
        z = this.post_quant_conv.forward(z);
        var dec = this.decoder.forward(z);

        return dec;
    }

    public Tensor decode(Tensor z)
    {
        var dec = this._decode(z);
        return dec;
    }

    public override Tensor forward(Tensor sample, bool sample_posterior = false, Generator? generator = null)
    {
        var x = sample;
        var posterior = this.encode(x);
        Tensor z;
        if (sample_posterior)
        {
            z = posterior.Sample(generator);
        }
        else
        {
            z = posterior.Mode();
        }

        var dec = this._decode(z);

        return dec;
    }

    public static AutoencoderKL FromPretrained(
        string pretrainedModelNameOrPath,
        string configName = "config.json",
        string modelWeightName = "diffusion_pytorch_model",
        bool useSafeTensor = true,
        ScalarType torchDtype = ScalarType.Float32
    )
    {
        var configPath = Path.Combine(pretrainedModelNameOrPath, configName);
        var json = File.ReadAllText(configPath);
        var config = JsonSerializer.Deserialize<Config>(json) ?? throw new ArgumentNullException("config");
        // if dtype is fp16, default FromDeprecatedAttnBlock to false
        if (torchDtype == ScalarType.Float16)
        {
            config.DecoderMidBlockFromDeprecatedAttnBlock = false;
            config.EncoderMidBlockAddAttention = false;
        }
        var autoEncoderKL = new AutoencoderKL(config, torchDtype);

        modelWeightName = (useSafeTensor, torchDtype) switch
        {
            (true, ScalarType.Float32) => $"{modelWeightName}.safetensors",
            (true, ScalarType.Float16) => $"{modelWeightName}.fp16.safetensors",
            (false, ScalarType.Float32) => $"{modelWeightName}.bin",
            (false, ScalarType.Float16) => $"{modelWeightName}.fp16.bin",
            _ => throw new ArgumentException("Invalid arguments for useSafeTensor and torchDtype")
        };

        

        var location = Path.Combine(pretrainedModelNameOrPath, modelWeightName);

        var loadedParameters = new Dictionary<string, bool>();
        autoEncoderKL.load_safetensors(location, strict: false, loadedParameters: loadedParameters);

        return autoEncoderKL;
    }

    public AutoencoderKL LoadFromModelConfig(
        string pretrainedModelNameOrPath,
        string configName = "config.json",
        string modelWeightName = "diffusion_pytorch_model",
        bool useSafeTensor = true,
        ScalarType torchDtype = ScalarType.Float32)
    {
        return AutoencoderKL.FromPretrained(pretrainedModelNameOrPath, configName, modelWeightName, useSafeTensor, torchDtype);
    }
}