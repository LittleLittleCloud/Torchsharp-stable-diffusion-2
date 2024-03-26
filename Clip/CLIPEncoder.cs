using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class CLIPEncoder : Module<Tensor, Tensor?, Tensor?, bool?, bool?, BaseModelOutput>
{
    private readonly CLIPTextConfig config;
    private readonly ModuleList<CLIPEncoderLayer> layers;
    private readonly bool gradient_checkpointing = false;

    public CLIPEncoder(CLIPTextConfig config)
        : base(nameof(CLIPEncoder))
    {
        this.config = config;
        this.layers = new ModuleList<CLIPEncoderLayer>(Enumerable.Range(0, config.NumHiddenLayers).Select(_ => new CLIPEncoderLayer(config)).ToArray());
        RegisterComponents();
    }

    public override BaseModelOutput forward(
        Tensor inputs_embeds,
        Tensor? attention_mask = null,
        Tensor? casual_attention_mask = null,
        bool? output_attentions = false,
        bool? output_hidden_states = false)
    {
        // inputs_embeds: [batch_size, seq_length, hidden_size]
        output_hidden_states = output_hidden_states ?? false;
        output_attentions = output_attentions ?? false;

        List<Tensor>? encoder_states = null;
        List<Tensor>? all_attentions = null;

        if (output_hidden_states is true)
        {
            encoder_states = new List<Tensor>();
        }

        if (output_attentions is true)
        {
            all_attentions = new List<Tensor>();
        }

        var hidden_states = inputs_embeds;
        foreach (var layer in layers)
        {
            if (encoder_states is not null)
            {
                encoder_states.Add(hidden_states);
            }
            (hidden_states, var attension_weight) = layer.forward(hidden_states, attention_mask, casual_attention_mask, output_attentions);

            if (all_attentions is not null && attension_weight is not null)
            {
                all_attentions.Add(attension_weight);
            }
        }

        if (encoder_states is not null)
        {
            encoder_states.Add(hidden_states);
        }

        return new BaseModelOutput(lastHiddenState: hidden_states, hiddenStates: encoder_states?.ToArray(), attentions: all_attentions?.ToArray());
    }
}