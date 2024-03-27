using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using System.Text.Json;
using TorchSharp.PyBridge;

namespace SD;

public class CLIPTextModel : Module<Tensor, Tensor?, Tensor?, bool?, bool?, BaseModelOutputWithPooling>, IModelConfigLoader<CLIPTextModel>
{
    private readonly CLIPTextConfig config;
    private readonly CLIPTextTransformer text_model;

    public CLIPTextModel(CLIPTextConfig config)
        : base(nameof(CLIPTextModel))
    {
        this.config = config;
        this.text_model = new CLIPTextTransformer(config);
        this.PostInit();
        RegisterComponents();
    }

    private void PostInit()
    {
        var factor = this.config.InitializerFactor;
    }

    public override BaseModelOutputWithPooling forward(
        Tensor input_ids,
        Tensor? attention_mask = null,
        Tensor? position_ids = null,
        bool? output_hidden_states = false,
        bool? output_attentions = false)
    {
        return this.text_model.forward(input_ids, attention_mask, position_ids, output_hidden_states, output_attentions);
    }

    public static CLIPTextModel FromPretrained(
        string pretrainedModelNameOrPath,
        string configName = "config.json",
        string modelWeightName = "model",
        bool useSafeTensor = true,
        ScalarType torchDtype = ScalarType.Float32)
    {
        var configPath = Path.Combine(pretrainedModelNameOrPath, configName);
        var json = File.ReadAllText(configPath);
        var config = JsonSerializer.Deserialize<CLIPTextConfig>(json) ?? throw new ArgumentNullException(nameof(CLIPTextConfig));

        var clipTextModel = new CLIPTextModel(config);

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
        clipTextModel.load_safetensors(location, strict: false, loadedParameters: loadedParameters);

        return clipTextModel;
    }

    public CLIPTextModel LoadFromModelConfig(
        string pretrainedModelNameOrPath,
        string configName = "config.json",
        string modelWeightName = "diffusion_pytorch_model",
        bool useSafeTensor = true,
        ScalarType torchDtype = ScalarType.Float32)
    {
        return CLIPTextModel.FromPretrained(pretrainedModelNameOrPath, configName, modelWeightName, useSafeTensor, torchDtype);
    }
}