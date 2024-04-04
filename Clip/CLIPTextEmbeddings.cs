using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace SD;

public class CLIPTextEmbeddings : Module<Tensor?, Tensor?, Tensor?, Tensor>
{
    private readonly CLIPTextConfig config;
    private readonly Embedding token_embedding;
    private readonly Embedding position_embedding;

    public CLIPTextEmbeddings(CLIPTextConfig config)
        : base(nameof(CLIPTextEmbeddings))
    {
        this.config = config;
        var embed_dim = config.HiddenSize;
        token_embedding = Embedding(config.VocabSize, embed_dim, dtype: config.DType);
        position_embedding = Embedding(config.MaxPositionEmbeddings, embed_dim, dtype: config.DType);

        this.register_buffer("position_ids", arange(config.MaxPositionEmbeddings).expand(1, -1), persistent: false);

        RegisterComponents();
    }

    public override Tensor forward(
        Tensor? input_ids = null,
        Tensor? position_ids = null,
        Tensor? inputs_embeds = null)
    {
        if (input_ids is null && position_ids is null && inputs_embeds is null)
        {
            throw new ArgumentException("You have to specify either input_ids or inputs_embeds");
        }
        var seq_length = input_ids is not null ? input_ids.shape[^1] : inputs_embeds!.shape[^2];
        var device = input_ids?.device ?? position_ids?.device ?? inputs_embeds?.device ?? throw new ArgumentException("You have to specify either input_ids or inputs_embeds");
        if (position_ids is null)
        {
            position_ids = this.get_buffer("position_ids")[.., ..(int)seq_length];
            position_ids = position_ids.to(device);
        }

        if (inputs_embeds is null)
        {
            inputs_embeds = this.token_embedding.forward(input_ids!);
        }

        var position_embeds = this.position_embedding.forward(position_ids);
        return inputs_embeds + position_embeds;
    }
}