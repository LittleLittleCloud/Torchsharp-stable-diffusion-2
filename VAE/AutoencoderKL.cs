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

    public AutoencoderKL(
        int in_channels = 3,
        int out_channels = 3,
        string[]? down_block_types = null,
        string[]? up_block_types = null,
        int[]? block_out_channels = null,
        int layers_per_block = 1,
        string act_fn = "silu",
        int latent_channels = 4,
        int norm_num_groups = 32,
        int sample_size = 32,
        float scaling_factor = 0.18215f,
        float[]? latents_mean = null,
        float[]? latents_std = null,
        bool force_upcast = true)
        : base(nameof(AutoencoderKL))
    {
        down_block_types = down_block_types ?? new string[] { "DownEncoderBlock2D" };
        up_block_types = up_block_types ?? new string[] { "UpDecoderBlock2D" };
        block_out_channels = block_out_channels ?? new int[] { 64 };

        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.down_block_types = down_block_types;
        this.up_block_types = up_block_types;
        this.block_out_channels = block_out_channels;
        this.layers_per_block = layers_per_block;
        this.act_fn = act_fn;
        this.latent_channels = latent_channels;
        this.norm_num_groups = norm_num_groups;
        this.sample_size = sample_size;
        this.scaling_factor = scaling_factor;
        this.latents_mean = latents_mean;
        this.latents_std = latents_std;
        this.force_upcast = force_upcast;

        this.encoder = new Encoder(
            inChannels: in_channels,
            outChannels: latent_channels,
            downBlockTypes: down_block_types,
            blockOutChannels: block_out_channels,
            layersPerBlock: layers_per_block,
            activationFunction: act_fn,
            normNumGroups: norm_num_groups,
            doubleZ: true);

        this.decoder = new Decoder(
            in_channels: latent_channels,
            out_channels: out_channels,
            up_block_types: up_block_types,
            block_out_channels: block_out_channels,
            layers_per_block: layers_per_block,
            norm_num_groups: norm_num_groups,
            act_fn: act_fn,
            mid_block_add_attention: true);

        this.latents_mean = latents_mean;
        this.latents_std = latents_std;
        this.force_upcast = force_upcast;

        this.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernelSize: 1, padding: Padding.Valid);
        this.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernelSize: 1, padding: Padding.Same);

        RegisterComponents();
    }

    public Decoder Decoder => this.decoder;

    public Encoder Encoder => this.encoder;

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
        var config = JsonSerializer.Deserialize<Config>(json);

        var autoEncoderKL = new AutoencoderKL(
            in_channels: config!.InChannels,
            out_channels: config!.OutChannels,
            down_block_types: config!.DownBlockTypes,
            up_block_types: config!.UpBlockTypes,
            block_out_channels: config!.BlockOutChannels,
            layers_per_block: config!.LayersPerBlock,
            act_fn: config!.ActivationFunction,
            latent_channels: config!.LatentChannels,
            norm_num_groups: config!.NormNumGroups,
            sample_size: config!.SampleSize);

        modelWeightName = (useSafeTensor, torchDtype) switch
        {
            (true, ScalarType.Float32) => $"{modelWeightName}.safetensors",
            (true, ScalarType.BFloat16) => $"{modelWeightName}.fp16.safetensors",
            (false, ScalarType.Float32) => $"{modelWeightName}.bin",
            (false, ScalarType.BFloat16) => $"{modelWeightName}.fp16.bin",
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