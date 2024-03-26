using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class CLIPTextModel : Module<Tensor, Tensor?, Tensor?, bool?, bool?, BaseModelOutputWithPooling>
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
}