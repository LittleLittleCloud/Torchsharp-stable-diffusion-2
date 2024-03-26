using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class BaseModelOutput
{
    public BaseModelOutput(
        Tensor lastHiddenState,
        Tensor[]? hiddenStates = null,
        Tensor[]? attentions = null)
    {
        LastHiddenState = lastHiddenState;
        HiddenStates = hiddenStates;
        Attentions = attentions;
    }

    public Tensor LastHiddenState { get; }

    public Tensor[]? HiddenStates { get; }

    public Tensor[]? Attentions { get; }
}

public class BaseModelOutputWithPooling : BaseModelOutput
{
    public BaseModelOutputWithPooling(
        Tensor lastHiddenState,
        Tensor poolerOutput,
        Tensor[]? hiddenStates = null,
        Tensor[]? attentions = null)
        : base(lastHiddenState, hiddenStates, attentions)
    {
        PoolerOutput = poolerOutput;
    }

    public Tensor PoolerOutput { get; }
}
public class CLIPTextTransformer : Module<Tensor, Tensor?, Tensor?, bool?, bool?, BaseModelOutputWithPooling>
{
    private readonly CLIPTextConfig config;
    private readonly CLIPTextEmbeddings embeddings;
    private readonly CLIPEncoder encoder;
    private readonly LayerNorm final_layer_norm;
    private readonly int eos_token_id;

    public CLIPTextTransformer(CLIPTextConfig config)
        : base(nameof(CLIPTextTransformer))
    {
        this.config = config;
        this.embeddings = new CLIPTextEmbeddings(config);
        this.encoder = new CLIPEncoder(config);
        this.final_layer_norm = LayerNorm(config.HiddenSize, eps: config.LayerNormEps);
        this.eos_token_id = config.EosTokenId;

        RegisterComponents();
    }

    public override BaseModelOutputWithPooling forward(
        Tensor input_ids,
        Tensor? attention_mask = null,
        Tensor? position_ids = null,
        bool? output_attentions = false,
        bool? output_hidden_states = false)
    {
        output_attentions = output_attentions ?? false;
        output_hidden_states = output_hidden_states ?? false;

        var input_shape = input_ids.shape;
        input_ids = input_ids.view(-1, input_shape[^1]);
        var hidden_states = this.embeddings.forward(input_ids: input_ids, position_ids: position_ids);
        var casual_attention_mask = AttentionMaskConverter.Create4DCasualAttentionMask(input_shape, hidden_states.dtype, hidden_states.device);

        if (attention_mask is not null)
        {
            attention_mask = AttentionMaskConverter.ExpandMask(attention_mask, hidden_states.dtype);
        }

        var encoder_outputs = this.encoder.forward(hidden_states, attention_mask, casual_attention_mask, output_attentions, output_hidden_states);

        var last_hidden_state = encoder_outputs.LastHiddenState;
        last_hidden_state = this.final_layer_norm.forward(last_hidden_state);
        Tensor pooled_output;
        if (this.eos_token_id == 2)
        {
            // The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            // A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            // ------------------------------------------------------------
            // text_embeds.shape = [batch_size, sequence_length, transformer.width]
            // take features from the eot embedding (eot_token is the highest number in each sequence)
            // casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device: last_hidden_state.device),
                input_ids.to(ScalarType.Int32).to(last_hidden_state.device).argmax(dim: 1)
            ];
        }
        else
        {
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device: last_hidden_state.device),
                (input_ids.to(ScalarType.Int32).to(last_hidden_state.device) == this.eos_token_id).to(ScalarType.Int32).argmax(dim: -1)
            ];
        }

        return new BaseModelOutputWithPooling(last_hidden_state, pooled_output, encoder_outputs.HiddenStates, encoder_outputs.Attentions);
    }
}