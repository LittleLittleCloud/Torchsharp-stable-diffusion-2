using System.Reflection.PortableExecutable;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

public class TokenizeDecoder : Microsoft.ML.Tokenizers.TokenizerDecoder
{
    private const char spaceReplacement = 'Ġ';

    private const char newlineReplacement = 'Ċ';

    private const char carriageReturnReplacement = 'č';
    private string bos = "<s>";
    private string eos = "</s>";

    public TokenizeDecoder(string bos = "<s>", string eos = "</s>")
    {
        this.bos = bos;
        this.eos = eos;
    }

    public override string Decode(IEnumerable<string> tokens)
    {
        var str = string.Join("", tokens);
        str = str.Replace(spaceReplacement, ' ');
        str = str.Replace(newlineReplacement, '\n');
        str = str.Replace(carriageReturnReplacement.ToString(), Environment.NewLine);

        if (str.StartsWith(bos))
        {
            str = str.Substring(bos.Length);
        }

        if (str.EndsWith(eos))
        {
            str = str.Substring(0, str.Length - eos.Length);
        }

        return str;
    }
}

public class BPETokenizer
{
    private Tokenizer tokenizer;
    private bool addPrecedingSpace;

    public BPETokenizer(
        string vocabPath,
        string mergesPath,
        bool addPrecedingSpace,
        string uknToken,
        string bosToken,
        string eosToken)
    {
        this.addPrecedingSpace = addPrecedingSpace;
        var bpe = new Bpe(vocabPath, mergesPath, endOfWordSuffix: "</w>");
        this.tokenizer = new Tokenizer(bpe);
        this.BosId = this.tokenizer.Model.TokenToId(bosToken) ?? throw new Exception("Failed to get bos id");
        this.EosId = this.tokenizer.Model.TokenToId(eosToken) ?? throw new Exception("Failed to get eos id");
        var decoder = new TokenizeDecoder(this.tokenizer.Model.IdToToken(this.BosId)!, this.tokenizer.Model.IdToToken(this.EosId)!);
        this.tokenizer.Decoder = decoder;
    }

    public static BPETokenizer FromPretrained(
        string folder,
        string vocabFile = "vocab.json",
        string mergesFile = "merges.txt",
        string specialTokensFile = "special_tokens_map.json",
        bool addPrecedingSpace = false,
        string uknToken = "<|endoftext|>",
        string bosToken = "<|startoftext|>",
        string eosToken = "<|endoftext|>")
    {
        var vocabPath = Path.Combine(folder, vocabFile);
        var mergesPath = Path.Combine(folder, mergesFile);
        var specialTokenMapPath = Path.Combine(folder, specialTokensFile);

        Dictionary<string, string>? specialTokenMap = null;
        // if (File.Exists(Path.Combine(folder, specialTokensFile)))
        // {
        //     specialTokenMap = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(specialTokenMapPath)) ?? throw new Exception("Failed to load special token map");
        // }

        bosToken = specialTokenMap?.GetValueOrDefault("bos_token") ?? bosToken;
        eosToken = specialTokenMap?.GetValueOrDefault("eos_token") ?? eosToken;
        uknToken = specialTokenMap?.GetValueOrDefault("unk_token") ?? uknToken;

        return new BPETokenizer(vocabPath, mergesPath, addPrecedingSpace, uknToken, bosToken, eosToken);
    }

    public int VocabSize => this.tokenizer.Model.GetVocabSize();

    public int PadId { get; }

    public int BosId { get; }

    public int EosId { get; }

    public string Decode(int[] input)
    {
        var str = this.tokenizer.Decode(input) ?? throw new Exception("Failed to decode");
        if (this.addPrecedingSpace)
        {
            str = str.TrimStart();
        }

        return str;
    }

    public int TokenToId(string token)
    {
        return this.tokenizer.Model.TokenToId(token) ?? throw new Exception("Failed to get token id");
    }

    public int[] Encode(string input, bool bos = false, bool eos = false)
    {
        if (this.addPrecedingSpace)
        {
            input = " " + input;
        }
        var tokens = this.tokenizer.Encode(input).Ids.ToArray();
        if (bos)
        {
            tokens = new int[] { this.BosId }.Concat(tokens).ToArray();
        }
        if (eos)
        {
            tokens = tokens.Concat(new int[] { this.EosId }).ToArray();
        }
        return tokens;
    }
}
