using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace SD;

public class Decoder : Module<Tensor, Tensor?, Tensor>
{
    private readonly int in_channels;
    private readonly int out_channels;
    private readonly string[] up_block_types;
    private readonly int[] block_out_channels;
    private readonly int layers_per_block;
    private readonly int norm_num_groups;
    private readonly string act_fn;
    private readonly string norm_type;
    private readonly bool mid_block_add_attention;
    private readonly ScalarType dtype;

    private readonly Conv2d conv_in;
    private readonly Module conv_norm_out;
    private readonly Module<Tensor, Tensor> conv_act;
    private readonly Module<Tensor, Tensor> conv_out;
    private readonly UNetMidBlock2D mid_block;
    private readonly ModuleList<Module<Tensor, Tensor?, Tensor>> up_blocks;
    public Decoder(
        int in_channels = 3,
        int out_channels = 3,
        string[]? up_block_types = null,
        int[]? block_out_channels = null,
        int layers_per_block = 2,
        int norm_num_groups = 32,
        string act_fn = "silu",
        string norm_type = "group",
        bool mid_block_add_attention = true,
        bool mid_block_from_deprecated_attn_block = true,
        ScalarType dtype = ScalarType.Float32)
        : base(nameof(Decoder))
    {
        up_block_types = up_block_types ?? new string[] { nameof(UpDecoderBlock2D) };
        block_out_channels = block_out_channels ?? new int[] { 64 };
        this.dtype = dtype;
        this.in_channels = in_channels;
        this.out_channels = out_channels;
        this.up_block_types = up_block_types;
        this.block_out_channels = block_out_channels;
        this.layers_per_block = layers_per_block;
        this.norm_num_groups = norm_num_groups;
        this.act_fn = act_fn;
        this.norm_type = norm_type;
        this.mid_block_add_attention = mid_block_add_attention;

        this.conv_in = torch.nn.Conv2d(this.in_channels, this.block_out_channels[^1], kernelSize: 3, stride: 1, padding: 1, dtype: this.dtype);
        int? temb_channels = norm_type == "spatial" ? in_channels : null;

        // mid
        this.mid_block = new UNetMidBlock2D(
            in_channels: this.block_out_channels[^1],
            resnet_eps: 1e-6f,
            resnet_act_fn: act_fn,
            output_scale_factor: 1.0f,
            resnet_time_scale_shift: norm_type == "group" ? "default" : norm_type,
            attention_head_dim: this.block_out_channels[^1],
            resnet_groups: norm_num_groups,
            temb_channels: temb_channels,
            add_attention: mid_block_add_attention,
            from_deprecated_attn_block: mid_block_from_deprecated_attn_block,
            dtype: this.dtype);

        // up
        var reversed_block_out_channels = block_out_channels.Reverse().ToArray();
        var output_channel = reversed_block_out_channels[0];
        this.up_blocks = new ModuleList<Module<Tensor, Tensor?, Tensor>>();
        for (int i = 0; i < up_block_types.Length; i++)
        {
            var prev_output_channel = output_channel;
            output_channel = reversed_block_out_channels[i];

            var is_final_block = i == up_block_types.Length - 1;
            var up_block = new UpDecoderBlock2D(
                in_channels: prev_output_channel,
                out_channels: output_channel,
                add_upsample: !is_final_block,
                num_layers: layers_per_block + 1,
                resnet_eps: 1e-6f,
                resnet_act_fn: act_fn,
                resnet_groups: norm_num_groups,
                temb_channels: temb_channels,
                dtype: this.dtype);
            
            this.up_blocks.Add(up_block);
            prev_output_channel = output_channel;
        }

        // out
        if (norm_type == "spatial")
        {
            this.conv_norm_out = new SpatialNorm(block_out_channels[0], temb_channels ?? 512, dtype: this.dtype);
        }
        else
        {
            this.conv_norm_out = GroupNorm(num_channels: block_out_channels[0], num_groups: norm_num_groups, eps: 1e-6f, dtype: this.dtype);
        }

        this.conv_act = nn.SiLU();
        this.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernelSize: 3, padding: Padding.Same, dtype: this.dtype);

        RegisterComponents();
    }

    public override Tensor forward(Tensor sample, Tensor? latent_embeds = null)
    {
        sample = this.conv_in.forward(sample);
        var upscale_dtype = this.up_blocks[0].parameters().First().dtype;
        
        // middle
        var input = new UNetMidBlock2DInput(sample, latent_embeds);
        sample = this.mid_block.forward(input);
        sample = sample.to(upscale_dtype);

        // up
        foreach (var up_block in this.up_blocks)
        {
            sample = up_block.forward(sample, latent_embeds);
        }

        // post-process
        if (latent_embeds is null && this.conv_norm_out is Module<Tensor, Tensor> norm)
        {
            sample = norm.forward(sample);
        }
        else if (this.conv_norm_out is Module<Tensor, Tensor?, Tensor> norm1)
        {
            sample = norm1.forward(sample, latent_embeds);
        }
        else
        {
            throw new ArgumentException("Invalid norm type: " + this.conv_norm_out.GetType().Name);
        }

        sample = this.conv_act.forward(sample);
        sample = this.conv_out.forward(sample);

        return sample;
    }
}