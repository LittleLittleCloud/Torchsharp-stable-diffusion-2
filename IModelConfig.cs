using TorchSharp;
using static TorchSharp.torch;

public interface IModelConfigLoader<out T>
{
    T LoadFromModelConfig(
        string pretrainedModelNameOrPath,
        string configName = "config.json",
        string modelWeightName = "diffusion_pytorch_model",
        bool useSafeTensor = true,
        ScalarType torchDtype = ScalarType.Float32);
}