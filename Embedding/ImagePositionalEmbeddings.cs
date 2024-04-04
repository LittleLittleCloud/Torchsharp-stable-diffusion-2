using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class ImagePositionalEmbeddings : Module<Tensor, Tensor>
{
    private readonly Embedding emb;
    private readonly Embedding height_emb;
    private readonly Embedding width_emb;
    private readonly int height;
    private readonly int width;
    private readonly int num_embed;
    private readonly int embed_dim;

    public ImagePositionalEmbeddings(
        int num_embed,
        int height,
        int width,
        int embed_dim,
        ScalarType dtype = ScalarType.Float32
    ) : base(nameof(ImagePositionalEmbeddings))
    {
        this.height = height;
        this.width = width;
        this.num_embed = num_embed;
        this.embed_dim = embed_dim;

        this.emb = Embedding(num_embed, embed_dim, dtype: dtype);
        this.height_emb = Embedding(height, embed_dim, dtype: dtype);
        this.width_emb = Embedding(width, embed_dim, dtype: dtype);

        RegisterComponents();
    }

    public override Tensor forward(Tensor index)
    {
        var emb = this.emb.forward(index);

        var height_emb = this.height_emb.forward(torch.arange(this.height, device: index.device).view(1, this.height));

        // 1 x H x D -> 1 x H x 1 x D
        height_emb = height_emb.unsqueeze(2);

        var width_emb = this.width_emb.forward(torch.arange(this.width, device: index.device).view(1, this.width));

        // 1 x W x D -> 1 x 1 x W x D
        width_emb = width_emb.unsqueeze(1);

        var pos_emb = height_emb + width_emb;

        // 1 x H x W x D -> 1 x L xD
        pos_emb = pos_emb.view(1, this.height * this.width, -1);

        emb = emb + pos_emb[.., 0..(int)emb.shape[1], 0..];

        return emb;
    }
}
