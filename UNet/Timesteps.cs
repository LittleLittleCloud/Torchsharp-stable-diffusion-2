namespace SD;

public class Timesteps: Module<Tensor, Tensor>
{
    private readonly int num_channels;
    private readonly bool flip_sin_to_cos;
    private readonly float downscale_freq_shift;

    public Timesteps(int num_channels, bool flip_sin_to_cos, float downscale_freq_shift): base("Timesteps")
    {
        this.num_channels = num_channels;
        this.flip_sin_to_cos = flip_sin_to_cos;
        this.downscale_freq_shift = downscale_freq_shift;
    }

    public override Tensor forward(Tensor timesteps)
    {
        var t_emb = Utils.GetTimestepEmbedding(
            timesteps,
            num_channels,
            flip_sin_to_cos: flip_sin_to_cos,
            downscale_freq_shift: downscale_freq_shift
        );
        return t_emb;
    }
}