using static TorchSharp.torch.nn;
using static TorchSharp.torch;
using TorchSharp.Modules;
using Xunit;
using ApprovalTests;
using ApprovalTests.Reporters;
using ApprovalTests.Namers;
using TorchSharp;
using FluentAssertions;

namespace SD;

public class TokenizerTest
{
    [Fact]
    public void TokenizerTest1()
    {
        var tokenizerFolder = "/home/xiaoyuz/stable-diffusion-2/tokenizer";

        var tokenizer = BPETokenizer.FromPretrained(tokenizerFolder);
        var input = "a photo of a cat";
        var output = tokenizer.Encode(input, true, true);
        output.Should().BeEquivalentTo([49406,   320,  1125,   539,   320,  2368, 49407]);
    }
}