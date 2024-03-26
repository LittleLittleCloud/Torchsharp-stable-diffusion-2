using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class DiagonalGaussianDistribution
{
    private readonly Tensor parameters;
    private readonly bool deterministic;
    private readonly Tensor mean;
    private readonly Tensor logvar;
    private readonly Tensor std;
    private readonly Tensor var;
    public DiagonalGaussianDistribution(
        Tensor parameters,
        bool deterministic = false)
    {
        this.parameters = parameters;
        this.deterministic = deterministic;

        var chunks = torch.chunk(parameters, 2, dim: -1);
        this.mean = chunks[0];
        this.logvar = chunks[1];
        this.std = torch.exp(0.5f * this.logvar);
        this.var = torch.exp(this.logvar);

        if (deterministic)
        {
            this.std = torch.zeros_like(this.std, device: this.parameters.device, dtype: this.parameters.dtype);
            this.var = torch.zeros_like(this.var, device: this.parameters.device, dtype: this.parameters.dtype);
        }
    }

    public Tensor Sample(Generator? generator = null)
    {
        if (deterministic)
        {
            return mean;
        }

        return mean + std * torch.randn_like(mean);
    }

    public Tensor KL(DiagonalGaussianDistribution? other)
    {
        if (this.deterministic)
        {
            return torch.zeros_like(this.mean);
        }

        if (other is null)
        {
            return 0.5 * torch.sum(
                this.var + this.mean * this.mean - 1.0 - this.logvar,
                dim: [1, 2, 3]
            );
        }

        return 0.5 * torch.sum(
            this.var / other.var + (this.mean - other.mean).pow(2) / other.var - 1.0 - this.logvar + other.logvar,
            dim: [1, 2, 3]
        );
    }

    public Tensor NLL(Tensor sample, long[]? dims = null)
    {
        dims = dims ?? new long[] { 1, 2, 3 };

        if (deterministic)
        {
            return torch.zeros_like(this.mean);
        }

        var log2Pi = torch.tensor(2.0 * Math.PI, device: this.parameters.device, dtype: this.parameters.dtype);
        var nll = 0.5 * torch.sum(
            (sample - this.mean).pow(2) / this.var + this.logvar + log2Pi,
            dim: dims
        );

        return nll;
    }

    public Tensor Mode()
    {
        return mean;
    }
}